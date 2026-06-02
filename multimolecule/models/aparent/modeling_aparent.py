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

from multimolecule.modules import Criterion, HeadConfig

from ..modeling_outputs import SequencePredictorOutput
from .configuration_aparent import AparentConfig


class AparentPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AparentConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["AparentEncoder"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)


class AparentModel(AparentPreTrainedModel):
    """
    The bare APARENT model outputting the shared sequence representation together with the upstream isoform and
    cleavage predictions.

    Examples:
        >>> from multimolecule import AparentConfig, AparentModel, RnaTokenizer
        >>> config = AparentConfig()
        >>> model = AparentModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/aparent")
        >>> input = tokenizer("ACGUNACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([1, 256])
        >>> output["isoform_logits"].shape
        torch.Size([1, 1])
        >>> output["cleavage_logits"].shape
        torch.Size([1, 206])
    """

    def __init__(self, config: AparentConfig):
        super().__init__(config)
        self.embeddings = AparentEmbedding(config)
        self.encoder = AparentEncoder(config)
        self.isoform_decoder = nn.Linear(config.hidden_sizes[-1] + config.library_size, config.num_isoform_labels)
        self.cleavage_decoder = nn.Linear(config.hidden_sizes[-1] + config.library_size, config.num_cleavage_labels)
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
    ) -> AparentModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if isinstance(input_ids, NestedTensor):
            attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = self.encoder(embedding_output)

        batch_size = pooled_output.size(0)
        # The upstream default encoder feeds an all-zero library one-hot before the output
        # layers. It is a deterministic constant rebuilt here rather than stored in the
        # checkpoint; it cannot be a static buffer because its leading dimension is batch_size.
        library = torch.zeros(
            batch_size,
            self.config.library_size,
            device=pooled_output.device,
            dtype=pooled_output.dtype,
        )
        shared = torch.cat([pooled_output, library], dim=-1)
        isoform_logits = self.isoform_decoder(shared)
        cleavage_logits = self.cleavage_decoder(shared)

        return AparentModelOutput(
            pooler_output=pooled_output,
            isoform_logits=isoform_logits,
            cleavage_logits=cleavage_logits,
        )


class AparentForSequencePrediction(AparentPreTrainedModel):
    """
    APARENT model with a sequence-level prediction head.

    APARENT's primary sequence-level output is the alternative-polyadenylation isoform score. This wrapper exposes the
    converted upstream isoform decoder directly. The upstream positional cleavage distribution is intentionally not
    exposed by this head; it remains available on [`AparentModel`] as `cleavage_logits`.

    Examples:
        >>> import torch
        >>> from multimolecule import AparentConfig, AparentForSequencePrediction, RnaTokenizer
        >>> config = AparentConfig()
        >>> model = AparentForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/aparent")
        >>> input = tokenizer("ACGUNACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1.0]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MseLossBackward0>)
    """

    def __init__(self, config: AparentConfig):
        super().__init__(config)
        self.model = AparentModel(config)
        head_config = HeadConfig(config.head) if config.head is not None else HeadConfig()
        if head_config.num_labels is None:
            head_config.num_labels = config.num_isoform_labels
        if head_config.problem_type is None:
            head_config.problem_type = "regression"
        self.head_config = head_config
        self.criterion = Criterion(head_config)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_isoform_labels != 1:
            return [f"isoform_proportion_{index}" for index in range(self.config.num_isoform_labels)]
        return ["isoform_proportion"]

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

        logits = outputs.isoform_logits
        loss = self.criterion(logits, labels) if labels is not None else None

        return SequencePredictorOutput(loss=loss, logits=logits)

    def postprocess(self, outputs: SequencePredictorOutput | ModelOutput) -> Tensor:
        return torch.sigmoid(outputs["logits"])


class AparentEmbedding(nn.Module):
    """One-hot input projection for the fixed-length APARENT sequence input."""

    def __init__(self, config: AparentConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.sequence_length = config.sequence_length
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
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            valid = (input_ids >= 0) & (input_ids < self.vocab_size)
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype=dtype)
            if not valid.all():
                inputs_embeds = inputs_embeds * valid.unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        # APARENT consumes a fixed-length window; pad with zeros or trim to ``sequence_length``.
        length = inputs_embeds.size(1)
        if length < self.sequence_length:
            pad = torch.zeros(
                inputs_embeds.size(0),
                self.sequence_length - length,
                self.vocab_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
        elif length > self.sequence_length:
            inputs_embeds = inputs_embeds[:, : self.sequence_length, :]
        # (batch, vocab_size, sequence_length) for the 1D convolutional stack.
        return inputs_embeds.transpose(1, 2)


class AparentEncoder(nn.Module):
    """Two-layer convolutional stack followed by two fully connected layers."""

    def __init__(self, config: AparentConfig):
        super().__init__()
        self.act = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(config.vocab_size, config.conv1_channels, config.conv1_kernel_size)
        self.pool = nn.MaxPool1d(config.pool_size)
        self.conv2 = nn.Conv1d(config.conv1_channels, config.conv2_channels, config.conv2_kernel_size)

        conv1_out = config.sequence_length - config.conv1_kernel_size + 1
        pooled_out = conv1_out // config.pool_size
        conv2_out = pooled_out - config.conv2_kernel_size + 1
        # The upstream Keras Flatten runs over (length, channels); the extra ``+ 1`` is the
        # scalar distal-PAS input concatenated before the first dense layer.
        input_size = conv2_out * config.conv2_channels + 1

        self.dense1 = nn.Linear(input_size, config.hidden_sizes[0])
        self.dropout1 = nn.Dropout(config.dropouts[0])
        self.dense2 = nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1])
        self.dropout2 = nn.Dropout(config.dropouts[1])
        self.library_size = config.library_size

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.act(self.conv1(hidden_state))
        hidden_state = self.pool(hidden_state)
        hidden_state = self.act(self.conv2(hidden_state))
        # Upstream flattens the channels-last Keras tensor (length, channels). The torch
        # Conv1d output is (channels, length); transpose before flattening so the dense
        # layer sees the same element order as the original checkpoint.
        hidden_state = hidden_state.transpose(1, 2).reshape(hidden_state.size(0), -1)
        # The upstream default encoder feeds a constant distal-PAS scalar of 1.0. It is a
        # deterministic constant rebuilt here rather than stored in the checkpoint; it cannot
        # be a static buffer because its leading dimension is batch_size.
        distal_pas = torch.ones(hidden_state.size(0), 1, device=hidden_state.device, dtype=hidden_state.dtype)
        hidden_state = torch.cat([hidden_state, distal_pas], dim=-1)
        hidden_state = self.dropout1(self.act(self.dense1(hidden_state)))
        hidden_state = self.dropout2(self.act(self.dense2(hidden_state)))
        return hidden_state


@dataclass
class AparentModelOutput(ModelOutput):
    """
    Base class for outputs of the APARENT model.

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The shared sequence representation after the two fully connected layers. Consumed by the MultiMolecule
            sequence-prediction head.
        isoform_logits (`torch.FloatTensor` of shape `(batch_size, num_isoform_labels)`):
            Pre-sigmoid logits of the upstream alternative-polyadenylation isoform-proportion output.
        cleavage_logits (`torch.FloatTensor` of shape `(batch_size, num_cleavage_labels)`):
            Pre-softmax logits of the upstream positional cleavage distribution.
    """

    pooler_output: torch.FloatTensor | None = None
    isoform_logits: torch.FloatTensor | None = None
    cleavage_logits: torch.FloatTensor | None = None
