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
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import SequencePredictionHead, preserve_batch_norm_stats

from ..modeling_outputs import SequencePredictorOutput
from .configuration_basset import BassetConfig


class BassetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BassetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["BassetConvLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class BassetModel(BassetPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import BassetConfig, BassetModel, DnaTokenizer
        >>> config = BassetConfig()
        >>> model = BassetModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/basset")
        >>> input = tokenizer(["ACGT" * 150, "TGCA" * 150], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 1000])
    """

    def __init__(self, config: BassetConfig):
        super().__init__(config)
        self.embeddings = BassetEmbedding(config)
        self.encoder = BassetEncoder(config)
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
    ) -> BassetModelOutput:
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
        # The Basset encoder collapses the sequence dimension through its fully-connected layers, so the
        # final feature vector is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return BassetModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class BassetForSequencePrediction(BassetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import BassetConfig, BassetForSequencePrediction, DnaTokenizer
        >>> config = BassetConfig()
        >>> model = BassetForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/basset")
        >>> input = tokenizer(["ACGT" * 150, "TGCA" * 150], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2, 164)))
        >>> output["logits"].shape
        torch.Size([2, 164])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: BassetConfig):
        super().__init__(config)
        self.model = BassetModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"dnase_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        return [f"dnase_{index}" for index in range(self.config.num_labels)]

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

    def postprocess(self, outputs: Any) -> Tensor:
        return torch.sigmoid(outputs["logits"])


class BassetEmbedding(nn.Module):
    """One-hot embedding layer for Basset.

    Basset does not use learned word embeddings; it consumes a one-hot encoding of the four DNA nucleotides
    transposed into `(batch_size, vocab_size, sequence_length)` for the 1D convolution stack.
    """

    def __init__(self, config: BassetConfig):
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
                f"Basset expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class BassetEncoder(nn.Module):
    def __init__(self, config: BassetConfig):
        super().__init__()
        layers = []
        in_channels = config.vocab_size
        for index in range(config.num_conv_layers):
            layers.append(
                BassetConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=config.conv_channels[index],
                    kernel_size=config.conv_kernel_sizes[index],
                    pool_size=config.conv_pool_sizes[index],
                )
            )
            in_channels = config.conv_channels[index]
        self.layers = nn.ModuleList(layers)
        fc_layers = []
        in_features = self._fc_input_size(config)
        for out_features in config.fc_sizes:
            fc_layers.append(BassetFullyConnectedLayer(config, in_features, out_features))
            in_features = out_features
        self.fc_layers = nn.ModuleList(fc_layers)
        self.gradient_checkpointing = False

    @staticmethod
    def _fc_input_size(config: BassetConfig) -> int:
        # The released Torch7 checkpoint uses valid convolutions and ceil-mode max-pooling.
        length = config.sequence_length
        for index in range(config.num_conv_layers):
            length = length - config.conv_kernel_sizes[index] + 1
            length = math.ceil(length / config.conv_pool_sizes[index])
        return length * config.conv_channels[-1]

    def forward(self, hidden_state: Tensor) -> Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_state,
                    context_fn=lambda layer=layer: (nullcontext(), preserve_batch_norm_stats(layer)),
                )
            else:
                hidden_state = layer(hidden_state)
        hidden_state = hidden_state.flatten(1)
        for layer in self.fc_layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class BassetConvLayer(nn.Module):
    def __init__(
        self,
        config: BassetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.batch_norm = nn.BatchNorm1d(out_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.activation = ACT2FN[config.hidden_act]
        self.pool = nn.MaxPool1d(pool_size, ceil_mode=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.pool(hidden_state)
        return hidden_state


class BassetFullyConnectedLayer(nn.Module):
    def __init__(self, config: BassetConfig, in_features: int, out_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features, config.batch_norm_eps, config.batch_norm_momentum)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


@dataclass
class BassetModelOutput(ModelOutput):
    """
    Base class for outputs of the Basset backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Final feature vector produced by the Basset encoder.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Same tensor as `last_hidden_state`; Basset collapses the sequence dimension in its encoder.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple containing the one-hot embedding output and the final encoder feature vector.
        attentions:
            Always `None`; Basset is a convolutional model and has no attention layers. Provided for compatibility
            with the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


BassetPreTrainedModel._can_record_outputs = {"hidden_states": BassetEncoder}
