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

from contextlib import nullcontext
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
from .configuration_cpgenie import CpGenieConfig


class CpGeniePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CpGenieConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["CpGenieConvLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class CpGenieModel(CpGeniePreTrainedModel):
    """
    Examples:
        >>> from multimolecule import CpGenieConfig, CpGenieModel, DnaTokenizer
        >>> config = CpGenieConfig()
        >>> model = CpGenieModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/cpgenie")
        >>> input = tokenizer(["ACGT" * 250 + "A", "TGCA" * 250 + "T"], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 64])
    """

    def __init__(self, config: CpGenieConfig):
        super().__init__(config)
        self.embeddings = CpGenieEmbedding(config)
        self.encoder = CpGenieEncoder(config)
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
        # The CpGenie encoder collapses the sequence dimension through its fully-connected layers, so the
        # final feature vector is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class CpGenieForSequencePrediction(CpGeniePreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import CpGenieConfig, CpGenieForSequencePrediction, DnaTokenizer
        >>> config = CpGenieConfig()
        >>> model = CpGenieForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/cpgenie")
        >>> input = tokenizer(["ACGT" * 250 + "A", "TGCA" * 250 + "T"], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2,)))
        >>> output["logits"].shape
        torch.Size([2, 2])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: CpGenieConfig):
        super().__init__(config)
        self.model = CpGenieModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"methylation_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        if self.config.num_labels == 2:
            return ["unmethylated", "methylated"]
        return [f"methylation_{index}" for index in range(self.config.num_labels)]

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

    def postprocess(self, outputs: Any) -> Tensor:
        return torch.softmax(outputs["logits"], dim=-1)


class CpGenieEmbedding(nn.Module):
    """One-hot embedding layer for CpGenie.

    CpGenie does not use learned word embeddings; it consumes a one-hot encoding of the four DNA nucleotides
    transposed into `(batch_size, vocab_size, sequence_length)` for the 1D convolution stack.
    """

    def __init__(self, config: CpGenieConfig):
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
                raise ValueError("You have to specify input_ids when inputs_embeds is not provided")
            self._check_sequence_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            self._check_sequence_length(inputs_embeds.size(1))
            inputs_embeds = inputs_embeds.to(dtype)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        return inputs_embeds

    def _check_sequence_length(self, sequence_length: int):
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"CpGenie expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class CpGenieEncoder(nn.Module):
    def __init__(self, config: CpGenieConfig):
        super().__init__()
        layers = []
        in_channels = config.vocab_size
        for index in range(config.num_conv_layers):
            layers.append(
                CpGenieConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=config.conv_channels[index],
                    kernel_size=config.conv_kernel_sizes[index],
                    pool_size=config.conv_pool_sizes[index],
                    pool_stride=config.conv_pool_strides[index],
                )
            )
            in_channels = config.conv_channels[index]
        self.layers = nn.ModuleList(layers)
        self.flattened_size = self._flattened_size(config)
        fc_layers = []
        in_features = self.flattened_size
        for out_features in config.fc_sizes:
            fc_layers.append(CpGenieFullyConnectedLayer(config, in_features, out_features))
            in_features = out_features
        self.fc_layers = nn.ModuleList(fc_layers)
        self.gradient_checkpointing = False

    @staticmethod
    def _flattened_size(config: CpGenieConfig) -> int:
        # Upstream CpGenie uses Keras `Convolution2D(..., border_mode='same')` (output length equals input
        # length) followed by valid (non-padded) `MaxPooling2D(pool_size=(1, k), strides=(1, s))`, so the
        # sequence shrinks only at the pooling stages: `length = floor((length - k) / s) + 1`.
        length = config.sequence_length
        for index in range(config.num_conv_layers):
            pool = config.conv_pool_sizes[index]
            stride = config.conv_pool_strides[index]
            length = (length - pool) // stride + 1
        return length * config.conv_channels[-1]

    def forward(self, hidden_state: Tensor) -> Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_state,
                    context_fn=lambda: (nullcontext(), nullcontext()),
                )
            else:
                hidden_state = layer(hidden_state)
        # Upstream Keras `Flatten` on a `(channels, 1, length)` feature map enumerates columns in
        # `(channel, length)` row-major order, which matches torch's `Tensor.flatten(1)` on a
        # `(channels, length)` tensor; no re-indexing is needed across the two frameworks.
        hidden_state = hidden_state.flatten(1)
        for layer in self.fc_layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class CpGenieConvLayer(nn.Module):
    def __init__(
        self,
        config: CpGenieConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        pool_stride: int,
    ):
        super().__init__()
        # Upstream CpGenie convolves with Keras `same` padding (output length equals input length) and a stride
        # of 1; replicate with torch `padding="same"` so the converted weights produce identical activations.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.activation = ACT2FN[config.hidden_act]
        # Keras `MaxPooling2D(pool_size=(1, k), strides=(1, s))` defaults to `valid` padding, which trims any
        # incomplete trailing window; torch `MaxPool1d` matches when `ceil_mode=False` (the default).
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.pool(hidden_state)
        return hidden_state


class CpGenieFullyConnectedLayer(nn.Module):
    def __init__(self, config: CpGenieConfig, in_features: int, out_features: int):
        super().__init__()
        # Upstream CpGenie applies dropout AFTER the fully-connected layer's activation, mirroring the
        # Keras Sequential ordering `Dense -> Activation -> Dropout`.
        self.dense = nn.Linear(in_features, out_features)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state
