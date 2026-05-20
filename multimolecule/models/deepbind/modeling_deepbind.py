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
from typing import Any, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import SequencePredictionHead

from ..modeling_outputs import SequencePredictorOutput
from .configuration_deepbind import DeepBindConfig


class DeepBindPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepBindConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["DeepBindEncoder"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)


class DeepBindModel(DeepBindPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DeepBindConfig, DeepBindModel
        >>> config = DeepBindConfig()
        >>> model = DeepBindModel(config)
        >>> output = model(torch.randint(4, (1, 101)))
        >>> output["pooler_output"].shape
        torch.Size([1, 32])
    """

    def __init__(self, config: DeepBindConfig):
        super().__init__(config)
        self.embeddings = DeepBindEmbedding(config)
        self.encoder = DeepBindEncoder(config)
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
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
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
        # The DeepBind encoder collapses the sequence dimension through global pooling, so the
        # pooled feature vector is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class DeepBindForSequencePrediction(DeepBindPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DeepBindConfig, DeepBindForSequencePrediction
        >>> config = DeepBindConfig()
        >>> model = DeepBindForSequencePrediction(config)
        >>> output = model(torch.randint(4, (1, 101)), labels=torch.tensor([[1.0]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: DeepBindConfig):
        super().__init__(config)
        self.model = DeepBindModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
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


class DeepBindEmbedding(nn.Module):
    """One-hot embedding layer for DeepBind.

    DeepBind does not use learned word embeddings; it consumes a one-hot encoding of the four
    nucleotides transposed into `(batch_size, vocab_size, sequence_length)` for the 1D convolution.
    Sequences may have arbitrary length: the downstream encoder pools globally over the sequence
    dimension, so different windows produce different feature vectors but use the same parameters.
    """

    def __init__(self, config: DeepBindConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.kernel_size = config.kernel_size
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
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
                raise ValueError("You have to specify input_ids when inputs_embeds is not provided")
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            self._check_sequence_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            self._check_sequence_length(inputs_embeds.size(1))
            inputs_embeds = inputs_embeds.to(dtype)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        return inputs_embeds

    def _check_sequence_length(self, sequence_length: int) -> None:
        if sequence_length < self.kernel_size:
            raise ValueError(
                f"DeepBind requires sequence length >= kernel_size ({self.kernel_size}); got {sequence_length}. "
                "Pad shorter inputs with the unknown / `N` token."
            )


class DeepBindEncoder(nn.Module):
    """Single Conv1D motif scan + global pooling + (optional) hidden FC + linear projection.

    Mirrors the released DeepBind tool's flat module sequence:
    ``Conv1d -> ReLU -> {GlobalMaxPool | concat(GlobalMaxPool, GlobalAvgPool)}
    -> [Dropout -> Linear -> ReLU] -> Linear``.

    With `hidden_size > 0` the hidden FC layer is included; with `hidden_size == 0` the pooled
    feature vector is projected directly to the binding score (the "no hidden unit" mode).
    """

    def __init__(self, config: DeepBindConfig):
        super().__init__()
        self.pooling = config.pooling
        self.conv = nn.Conv1d(config.vocab_size, config.num_filters, kernel_size=config.kernel_size, padding=0)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout) if config.hidden_dropout > 0 else nn.Identity()
        if config.num_hidden > 0:
            self.dense = nn.Linear(config.pooled_size, config.num_hidden)
            self.hidden_activation = ACT2FN[config.hidden_act]
        else:
            self.dense = None
            self.hidden_activation = None

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.activation(hidden_state)
        if self.pooling == "maxavg":
            pooled_max = hidden_state.amax(dim=-1)
            pooled_avg = hidden_state.mean(dim=-1)
            hidden_state = torch.cat([pooled_max, pooled_avg], dim=-1)
        else:
            hidden_state = hidden_state.amax(dim=-1)
        hidden_state = self.dropout(hidden_state)
        if self.dense is not None and self.hidden_activation is not None:
            hidden_state = self.dense(hidden_state)
            hidden_state = self.hidden_activation(hidden_state)
        return hidden_state
