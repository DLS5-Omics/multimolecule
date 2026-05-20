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

from dataclasses import dataclass
from typing import Any, cast

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

from multimolecule.modules import SequencePredictionHead

from ..modeling_outputs import SequencePredictorOutput
from .configuration_deepsea import DeepSeaConfig


class DeepSeaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepSeaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["DeepSeaConvLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)


class DeepSeaModel(DeepSeaPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import DeepSeaConfig, DeepSeaModel, DnaTokenizer
        >>> config = DeepSeaConfig()
        >>> model = DeepSeaModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepsea")
        >>> input = tokenizer(["ACGT" * 250, "TGCA" * 250], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 925])
    """

    def __init__(self, config: DeepSeaConfig):
        super().__init__(config)
        self.embeddings = DeepSeaEmbedding(config)
        self.encoder = DeepSeaEncoder(config)
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
    ) -> DeepSeaModelOutput:
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
        # The DeepSEA encoder collapses the sequence dimension through its fully-connected stack, so the
        # final feature vector is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return DeepSeaModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class DeepSeaForSequencePrediction(DeepSeaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DeepSeaConfig, DeepSeaForSequencePrediction, DnaTokenizer
        >>> config = DeepSeaConfig()
        >>> model = DeepSeaForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepsea")
        >>> input = tokenizer(["ACGT" * 250, "TGCA" * 250], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2, 919)))
        >>> output["logits"].shape
        torch.Size([2, 919])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: DeepSeaConfig):
        super().__init__(config)
        self.model = DeepSeaModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"chromatin_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        return [f"chromatin_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        if self.config.reverse_complement_average:
            input_ids, attention_mask, inputs_embeds = self._prepare_inputs(input_ids, attention_mask, inputs_embeds)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.sequence_head(outputs, labels=None)
        logits = output.logits

        if self.config.reverse_complement_average:
            reverse_outputs = self.model(
                self._reverse_complement_input_ids(input_ids) if input_ids is not None else None,
                attention_mask=attention_mask.flip(-1) if attention_mask is not None else None,
                inputs_embeds=(
                    self._reverse_complement_inputs_embeds(inputs_embeds) if inputs_embeds is not None else None
                ),
                return_dict=True,
                **kwargs,
            )
            reverse_output = self.sequence_head(reverse_outputs, labels=None)
            probabilities = (torch.sigmoid(logits) + torch.sigmoid(reverse_output.logits)) / 2
            probabilities = probabilities.clamp(min=1e-7, max=1.0 - 1e-7)
            logits = torch.logit(probabilities)

        loss = self._compute_loss(logits, labels)

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def postprocess(self, outputs: Any) -> Tensor:
        return torch.sigmoid(outputs["logits"])

    @staticmethod
    def _prepare_inputs(
        input_ids: Tensor | NestedTensor | None,
        attention_mask: Tensor | None,
        inputs_embeds: Tensor | NestedTensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor
        return input_ids, attention_mask, inputs_embeds

    @staticmethod
    def _reverse_complement_input_ids(input_ids: Tensor | None) -> Tensor | None:
        if input_ids is None:
            return None
        reverse_input_ids = input_ids.flip(-1)
        # Complement lookup under the MultiMolecule DNA alphabet where A=0, C=1, G=2, T=3 (nucleobase order):
        # A(0)↔T(3), C(1)↔G(2).  Token ids outside [0, 3] (e.g. N or padding) are left unchanged below.
        complement = torch.tensor([3, 2, 1, 0], device=input_ids.device, dtype=input_ids.dtype)
        valid = (reverse_input_ids >= 0) & (reverse_input_ids < complement.numel())
        complemented = complement[reverse_input_ids.clamp(min=0, max=complement.numel() - 1).long()]
        return torch.where(valid, complemented, reverse_input_ids)

    @staticmethod
    def _reverse_complement_inputs_embeds(inputs_embeds: Tensor | None) -> Tensor | None:
        if inputs_embeds is None:
            return None
        channels = torch.arange(inputs_embeds.size(-1) - 1, -1, -1, device=inputs_embeds.device)
        return inputs_embeds.flip(1).index_select(-1, channels)

    def _compute_loss(self, logits: Tensor, labels: Tensor | None) -> torch.FloatTensor | None:
        # Use sequence_head.criterion directly rather than calling sequence_head(outputs, labels) again:
        # when reverse_complement_average is True the logits already encode the averaged probability in
        # logit space (after both branches are run and merged), so the criterion must receive those final
        # re-logited values — not recompute them from the raw encoder output a second time.
        if labels is None:
            return None
        loss = self.sequence_head.criterion(logits, labels)
        loss_weight = getattr(self.sequence_head, "loss_weight", None)
        if loss_weight is not None:
            loss = loss * loss_weight
        return cast(torch.FloatTensor, loss)


class DeepSeaEmbedding(nn.Module):
    """One-hot embedding layer for DeepSEA.

    DeepSEA does not use learned word embeddings; it consumes a one-hot encoding of the four DNA nucleotides
    transposed into `(batch_size, vocab_size, sequence_length)` for the 1D convolution stack.
    """

    def __init__(self, config: DeepSeaConfig):
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
                f"DeepSEA expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class DeepSeaEncoder(nn.Module):
    def __init__(self, config: DeepSeaConfig):
        super().__init__()
        layers = []
        in_channels = config.vocab_size
        for index in range(config.num_conv_layers):
            layers.append(
                DeepSeaConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=config.conv_channels[index],
                    kernel_size=config.conv_kernel_sizes[index],
                    pool_size=config.conv_pool_sizes[index],
                    dropout=config.conv_dropouts[index],
                )
            )
            in_channels = config.conv_channels[index]
        self.layers = nn.ModuleList(layers)
        fc_layers = []
        in_features = self._fc_input_size(config)
        for out_features in config.fc_sizes:
            fc_layers.append(DeepSeaFullyConnectedLayer(config, in_features, out_features))
            in_features = out_features
        self.fc_layers = nn.ModuleList(fc_layers)
        self.gradient_checkpointing = False

    @staticmethod
    def _fc_input_size(config: DeepSeaConfig) -> int:
        # Upstream DeepSEA uses *valid* (zero-padding) convolutions followed by floor-mode max-pooling, so
        # the sequence is trimmed by ``kernel_size - 1`` at every conv and floor-divided by the pool size
        # at every pool. The third convolution is followed by a dropout instead of a pool (``pool_size=1``).
        length = config.sequence_length
        for index in range(config.num_conv_layers):
            length = length - config.conv_kernel_sizes[index] + 1
            length = length // config.conv_pool_sizes[index]
        return length * config.conv_channels[-1]

    def forward(self, hidden_state: Tensor) -> Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(layer.__call__, hidden_state)
            else:
                hidden_state = layer(hidden_state)
        hidden_state = hidden_state.flatten(1)
        for layer in self.fc_layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class DeepSeaConvLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeaConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ):
        super().__init__()
        # Upstream DeepSEA convolves with *valid* padding and pools with floor-mode max-pooling.
        # The third convolutional block omits the pool (``pool_size=1``) and applies its dropout
        # directly on the convolutional activations before the fully-connected stack.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.activation = ACT2FN[config.hidden_act]
        self.pool_size = pool_size
        self.pool: nn.Module
        if pool_size > 1:
            self.pool = nn.MaxPool1d(pool_size)
        else:
            self.pool = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.pool(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class DeepSeaFullyConnectedLayer(nn.Module):
    def __init__(self, config: DeepSeaConfig, in_features: int, out_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


@dataclass
class DeepSeaModelOutput(ModelOutput):
    """
    Base class for outputs of the DeepSEA backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Final feature vector produced by the DeepSEA encoder.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Same tensor as `last_hidden_state`; DeepSEA collapses the sequence dimension in its encoder.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple containing the one-hot embedding output and the final encoder feature vector.
        attentions:
            Always `None`; DeepSEA is a convolutional model and has no attention layers. Provided for compatibility
            with the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


DeepSeaPreTrainedModel._can_record_outputs = {"hidden_states": DeepSeaEncoder}
