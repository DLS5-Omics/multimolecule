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
from .configuration_xpresso import XpressoConfig


class XpressoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XpressoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["XpressoBlock"]

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


class XpressoModel(XpressoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import XpressoConfig, XpressoModel, DnaTokenizer
        >>> config = XpressoConfig()
        >>> model = XpressoModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/xpresso")
        >>> input = tokenizer(["ACGTACGTACGT", "TGCATGCATGCA"], return_tensors="pt")
        >>> features = torch.randn(2, config.num_features)
        >>> output = model(**input, features=features)
        >>> output["pooler_output"].shape
        torch.Size([2, 2])
    """

    def __init__(self, config: XpressoConfig):
        super().__init__(config)
        self.embeddings = XpressoEmbedding(config)
        self.encoder = XpressoEncoder(config)
        self.head = XpressoHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    # Xpresso's `last_hidden_state` is the *flattened* convolutional representation, not a
    # per-position layer output, so it must not be tied into the recorded `hidden_states` tuple.
    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        features: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> XpressoModelOutput:
        """
        Args:
            input_ids: Token ids of the promoter sequence.
            attention_mask: Binary mask; 1 for real tokens, 0 for padding.
            inputs_embeds: Pre-computed one-hot (or soft) embeddings. Mutually exclusive with
                `input_ids`.
            features: Optional auxiliary tensor of shape `(batch_size, config.num_features)`
                containing numeric mRNA half-life features (e.g. 3′-UTR length, Kozak score).
                Required when `config.num_features > 0`; must be `None` when
                `config.num_features == 0`. The tensor is concatenated with the flattened
                convolutional representation before the fully-connected head.
        """
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
        if input_ids is not None:
            batch_size = input_ids.size(0)
        else:
            if inputs_embeds is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            batch_size = inputs_embeds.size(0)
        self._validate_features(features, batch_size)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(embedding_output, **kwargs)
        conv_output = encoder_outputs.last_hidden_state
        pooler_output = self.head(conv_output, features=features)

        return XpressoModelOutput(
            last_hidden_state=conv_output,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=None,
        )

    def _validate_features(self, features: Tensor | None, batch_size: int) -> None:
        if self.config.num_features == 0:
            if features is not None:
                raise ValueError(
                    "This Xpresso model is configured with num_features=0 and does not accept a `features` tensor."
                )
            return
        if features is None:
            raise ValueError(
                f"This Xpresso model is configured with num_features={self.config.num_features}; "
                "you must pass the auxiliary `features` tensor."
            )
        if features.ndim != 2:
            raise ValueError(
                "`features` must be a 2D tensor of shape "
                f"(batch_size, {self.config.num_features}), got shape {tuple(features.shape)}."
            )
        if features.size(0) != batch_size:
            raise ValueError(f"`features` batch size ({features.size(0)}) must match input batch size ({batch_size}).")
        if features.size(1) != self.config.num_features:
            raise ValueError(
                f"`features` last dimension ({features.size(1)}) must equal "
                f"`config.num_features` ({self.config.num_features})."
            )


class XpressoForSequencePrediction(XpressoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import XpressoConfig, XpressoForSequencePrediction, DnaTokenizer
        >>> config = XpressoConfig()
        >>> model = XpressoForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/xpresso")
        >>> input = tokenizer(["ACGTACGTACGT", "TGCATGCATGCA"], return_tensors="pt")
        >>> features = torch.randn(2, config.num_features)
        >>> output = model(**input, features=features, labels=torch.randn(2, 1))
        >>> output["logits"].shape
        torch.Size([2, 1])
    """

    def __init__(self, config: XpressoConfig):
        super().__init__(config)
        self.model = XpressoModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_labels == 1:
            return ["expression"]
        return [f"expression_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        features: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            features=features,
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


class XpressoEmbedding(nn.Module):
    def __init__(self, config: XpressoConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.input_length = config.input_length
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
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        # Xpresso consumes a fixed promoter window: right-pad or center-crop to `input_length`.
        length = inputs_embeds.size(2)
        if length < self.input_length:
            pad = torch.zeros(
                inputs_embeds.size(0),
                self.vocab_size,
                self.input_length - length,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad], dim=2)
        elif length > self.input_length:
            start = (length - self.input_length) // 2
            inputs_embeds = inputs_embeds[:, :, start : start + self.input_length]
        return inputs_embeds


class XpressoEncoder(nn.Module):
    def __init__(self, config: XpressoConfig):
        super().__init__()
        self.config = config
        in_channels = config.vocab_size
        blocks = []
        for index in range(config.num_conv_layers):
            blocks.append(
                XpressoBlock(
                    config,
                    in_channels=in_channels,
                    out_channels=config.conv_channels[index],
                    kernel_size=config.conv_kernel_sizes[index],
                    dilation=config.conv_dilations[index],
                    pool_size=config.pool_sizes[index],
                )
            )
            in_channels = config.conv_channels[index]
        self.blocks = nn.ModuleList(blocks)
        self.hidden_state_recorder = XpressoHiddenStateRecorder()
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> XpressoEncoderOutput:
        record_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        for block in self.blocks:
            previous_hidden_state = hidden_state
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(block.__call__, hidden_state)
            else:
                hidden_state = block(hidden_state)
            if record_hidden_states:
                hidden_state = self.hidden_state_recorder(
                    previous_hidden_state.transpose(1, 2),
                    hidden_state.transpose(1, 2),
                ).transpose(1, 2)
        return XpressoEncoderOutput(
            last_hidden_state=torch.flatten(hidden_state, start_dim=1),
            hidden_states=None,
        )


class XpressoBlock(nn.Module):
    def __init__(
        self,
        config: XpressoConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        pool_size: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.act = ACT2FN[config.hidden_act]
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.pool(hidden_state)
        return hidden_state


class XpressoHead(nn.Module):
    def __init__(self, config: XpressoConfig):
        super().__init__()
        self.input_length = config.input_length
        self.num_features = config.num_features
        in_features = self._fc_input_size(config) + config.num_features
        layers = []
        for dim in config.fc_dims:
            layers.append(XpressoDense(config, in_features=in_features, out_features=dim))
            in_features = dim
        self.layers = nn.ModuleList(layers)
        self.out_features = in_features

    @staticmethod
    def _fc_input_size(config: XpressoConfig) -> int:
        length = config.input_length
        channels = config.vocab_size
        for index in range(config.num_conv_layers):
            channels = config.conv_channels[index]
            length = length // config.pool_sizes[index]
        return channels * length

    def forward(self, conv_output: Tensor, features: Tensor | None = None) -> Tensor:
        hidden_state = conv_output
        if self.num_features:
            if features is None:
                raise ValueError("`features` must be provided when `num_features` is non-zero.")
            features = features.to(device=hidden_state.device, dtype=hidden_state.dtype)
            hidden_state = torch.cat([hidden_state, features], dim=1)
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class XpressoDense(nn.Module):
    def __init__(self, config: XpressoConfig, in_features: int, out_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.act = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class XpressoHiddenStateRecorder(nn.Module):
    """Identity module whose forward output is captured by the Transformers output-recording hooks."""

    def forward(self, previous_hidden_state: Tensor, hidden_state: Tensor) -> Tensor:
        return hidden_state


@dataclass
class XpressoEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class XpressoModelOutput(ModelOutput):
    """
    Base class for outputs of the Xpresso backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, flattened_conv_size)`):
            Flattened convolutional representation of the promoter sequence.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Final fully-connected representation, with the auxiliary mRNA half-life features fused in. This is the
            tensor consumed by [`SequencePredictionHead`][multimolecule.modules.SequencePredictionHead].
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the embedding output plus one after each convolutional block) of
            shape `(batch_size, length, channels)`. Convolutional feature maps recorded along the encoder stack.
        attentions (always `None`):
            Xpresso is a purely convolutional architecture and has no attention; this field is always `None` and is
            present only for compatibility with the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


XpressoPreTrainedModel._can_record_outputs = {
    "hidden_states": XpressoHiddenStateRecorder,
}
