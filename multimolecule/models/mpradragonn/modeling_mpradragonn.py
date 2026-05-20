# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import SequencePredictionHead, preserve_batch_norm_stats

from ..modeling_outputs import SequencePredictorOutput
from .configuration_mpradragonn import MpraDragoNnConfig


class MpraDragoNnPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MpraDragoNnConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["MpraDragoNnBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class MpraDragoNnModel(MpraDragoNnPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import MpraDragoNnConfig, MpraDragoNnModel, DnaTokenizer
        >>> config = MpraDragoNnConfig()
        >>> model = MpraDragoNnModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/mpradragonn")
        >>> input = tokenizer(["ACGT" * 36 + "A", "TGCA" * 36 + "T"], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 15960])
    """

    def __init__(self, config: MpraDragoNnConfig):
        super().__init__(config)
        self.embeddings = MpraDragoNnEmbedding(config)
        self.encoder = MpraDragoNnEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MpraDragoNnModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = self.encoder(embedding_output)

        # MPRA-DragoNN has no learned pooler: the flattened convolutional feature map is fed
        # directly into the Dense regression head. Expose it as both `last_hidden_state` and
        # `pooler_output` so the shared `SequencePredictionHead` (which reads `pooler_output`)
        # consumes the right tensor without an extra projection.
        return MpraDragoNnModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class MpraDragoNnForSequencePrediction(MpraDragoNnPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import MpraDragoNnConfig, MpraDragoNnForSequencePrediction, DnaTokenizer
        >>> config = MpraDragoNnConfig()
        >>> model = MpraDragoNnForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/mpradragonn")
        >>> input = tokenizer(["ACGT" * 36 + "A", "TGCA" * 36 + "T"], return_tensors="pt")
        >>> output = model(**input, labels=torch.randn(2, 12))
        >>> output["logits"].shape
        torch.Size([2, 12])
    """

    def __init__(self, config: MpraDragoNnConfig):
        super().__init__(config)
        self.model = MpraDragoNnModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        # Upstream Sharpr-MPRA task order from the Kipoi ConvModel schema:
        #   k562 minP {rep1, rep2, avg}, k562 sv40p {rep1, rep2, avg},
        #   hepg2 minP {rep1, rep2, avg}, hepg2 sv40p {rep1, rep2, avg}.
        if self.config.num_labels == 12:
            return [
                "k562_minp_rep1",
                "k562_minp_rep2",
                "k562_minp_avg",
                "k562_sv40p_rep1",
                "k562_sv40p_rep2",
                "k562_sv40p_avg",
                "hepg2_minp_rep1",
                "hepg2_minp_rep2",
                "hepg2_minp_avg",
                "hepg2_sv40p_rep1",
                "hepg2_sv40p_rep2",
                "hepg2_sv40p_avg",
            ]
        return [f"mpra_activity_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MpraDragoNnEmbedding(nn.Module):
    def __init__(self, config: MpraDragoNnConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.input_length = config.input_length
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self._dtype_reference: Tensor
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be specified when inputs_embeds is not provided")
            self._check_input_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            self._check_input_length(inputs_embeds.size(1))
            inputs_embeds = inputs_embeds.to(dtype)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)

    def _check_input_length(self, input_length: int):
        if input_length != self.input_length:
            raise ValueError(
                f"MPRA-DragoNN expects fixed-length {self.input_length} bp inputs, but got {input_length}. "
                "Pad or crop the sequence to match the configured input_length."
            )


class MpraDragoNnEncoder(nn.Module):
    def __init__(self, config: MpraDragoNnConfig):
        super().__init__()
        in_channels = config.vocab_size
        blocks = []
        for out_channels, kernel_size in zip(config.conv_channels, config.conv_kernel_sizes):
            blocks.append(MpraDragoNnBlock(config, in_channels, out_channels, kernel_size))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.flatten = nn.Flatten(start_dim=1)
        self.gradient_checkpointing = False

    def forward(self, hidden_state: Tensor) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_state,
                    context_fn=lambda block=block: (nullcontext(), preserve_batch_norm_stats(block)),
                )
            else:
                hidden_state = block(hidden_state)
        return self.flatten(hidden_state)


class MpraDragoNnBlock(nn.Module):
    def __init__(self, config: MpraDragoNnConfig, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        # Upstream Keras Conv1D uses `padding='valid'` (no padding), trimming `kernel_size - 1`
        # samples per layer. The ReLU activation is fused into the Keras Conv1D op upstream; we
        # apply BatchNorm before the activation (the Keras Sequential order is
        # Conv -> BN -> Dropout because the activation lives inside the Conv1D layer, so BN
        # actually sees the post-ReLU feature map). Replicate that order explicitly here.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.act = ACT2FN[config.hidden_act]
        self.norm = nn.BatchNorm1d(out_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


@dataclass
class MpraDragoNnModelOutput(ModelOutput):
    """
    Base class for outputs of the MPRA-DragoNN model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, pooled_length * conv_channels[-1])`):
            Flattened feature map produced by the convolutional encoder.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, pooled_length * conv_channels[-1])`):
            Sequence-level representation. MPRA-DragoNN has no learned pooler, so this is the same flattened
            convolutional feature map as `last_hidden_state`; the regression head consumes it directly.
        hidden_states:
            Always `None`; MPRA-DragoNN feature maps are not recorded.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Always `None`; MPRA-DragoNN is a convolutional model without attention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
