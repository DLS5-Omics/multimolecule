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

from multimolecule.modules import SequencePredictionHead

from ..modeling_outputs import SequencePredictorOutput
from .configuration_optmrl import OptMrlConfig


class OptMrlPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OptMrlConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["OptMrlEncoder"]

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


class OptMrlModel(OptMrlPreTrainedModel):
    """
    The bare OptMRL model outputting the dense bottleneck representation used by the regression head.

    Examples:
        >>> import torch
        >>> from multimolecule import OptMrlConfig, OptMrlModel
        >>> config = OptMrlConfig()
        >>> model = OptMrlModel(config)
        >>> output = model(torch.randint(config.vocab_size, (1, config.sequence_length)))
        >>> output["pooler_output"].shape
        torch.Size([1, 40])
    """

    def __init__(self, config: OptMrlConfig):
        super().__init__(config)
        self.embeddings = OptMrlEmbedding(config)
        self.encoder = OptMrlEncoder(config)
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
    ) -> OptMrlModelOutput:
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

        pooled_output = self.encoder(embedding_output)

        return OptMrlModelOutput(pooler_output=pooled_output)


class OptMrlForSequencePrediction(OptMrlPreTrainedModel):
    """
    OptMRL model with a sequence-level prediction head emitting the mean ribosome load (MRL) scalar.

    Examples:
        >>> import torch
        >>> from multimolecule import OptMrlConfig, OptMrlForSequencePrediction
        >>> config = OptMrlConfig()
        >>> model = OptMrlForSequencePrediction(config)
        >>> output = model(
        ...     torch.randint(config.vocab_size, (1, config.sequence_length)),
        ...     labels=torch.tensor([[1.0]]),
        ... )
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MseLossBackward0>)
    """

    def __init__(self, config: OptMrlConfig):
        super().__init__(config)
        self.model = OptMrlModel(config)
        self.sequence_head = SequencePredictionHead(config, config.head)
        self.head_config = self.sequence_head.config
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        num_labels = int(self.sequence_head.num_labels)
        if num_labels != 1:
            return [f"mean_ribosome_load_{index}" for index in range(num_labels)]
        return ["mean_ribosome_load"]

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

        return SequencePredictorOutput(loss=output.loss, logits=output.logits)


class OptMrlEmbedding(nn.Module):
    """One-hot input projection for the fixed-length 50 nt 5'UTR window."""

    def __init__(self, config: OptMrlConfig):
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
        # OptMRL consumes a fixed-length window; pad with zeros or trim to ``sequence_length``.
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


class OptMrlEncoder(nn.Module):
    """Three-layer 1D convolutional stack followed by a dense bottleneck."""

    def __init__(self, config: OptMrlConfig):
        super().__init__()
        self.act = ACT2FN[config.hidden_act]
        # Keras `padding="same"` with stride=1 pads `(k - 1) // 2` on the left and `k // 2` on the right.
        # PyTorch's `Conv1d(padding="same")` does the opposite for even kernels, so explicit padding
        # is used to match the upstream behaviour exactly.
        self.pad = nn.ConstantPad1d(((config.conv_kernel_size - 1) // 2, config.conv_kernel_size // 2), 0.0)

        self.conv_layers = nn.ModuleList()
        self.conv_dropouts = nn.ModuleList()
        in_channels = config.vocab_size
        for index in range(config.num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    config.conv_channels,
                    kernel_size=config.conv_kernel_size,
                )
            )
            # Upstream applies dropout after the second and third convolutions only.
            dropout = config.conv_dropout if index >= 1 else 0.0
            self.conv_dropouts.append(nn.Dropout(dropout))
            in_channels = config.conv_channels

        input_size = config.sequence_length * config.conv_channels
        # Upstream applies a linear `Dense` followed by a separate `Activation('relu')` layer.
        # The split is preserved here so the conversion stays a one-to-one weight mapping.
        self.dense = nn.Linear(input_size, config.hidden_size)
        self.dense_dropout = nn.Dropout(config.dense_dropout)
        self.conv_kernel_size = config.conv_kernel_size
        self.sequence_length = config.sequence_length

    def forward(self, hidden_state: Tensor) -> Tensor:
        for conv, dropout in zip(self.conv_layers, self.conv_dropouts):
            hidden_state = self.pad(hidden_state)
            hidden_state = conv(hidden_state)
            hidden_state = self.act(hidden_state)
            hidden_state = dropout(hidden_state)
        # Upstream flattens the channels-last Keras tensor (length, channels). The torch Conv1d
        # output is (channels, length); transpose before flattening so the dense layer sees the
        # same element order as the original checkpoint.
        hidden_state = hidden_state.transpose(1, 2).reshape(hidden_state.size(0), -1)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dense_dropout(hidden_state)
        return hidden_state


@dataclass
class OptMrlModelOutput(ModelOutput):
    """
    Base class for outputs of the OptMRL model.

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The dense bottleneck representation consumed by the MultiMolecule sequence-prediction head.
    """

    pooler_output: torch.FloatTensor | None = None
