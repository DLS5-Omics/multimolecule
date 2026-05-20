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
from .configuration_deepcpgdna import DeepCpgDnaConfig


class DeepCpgDnaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepCpgDnaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["DeepCpgDnaConvLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        # Upstream uses Keras `glorot_uniform` (the default) for both convolutions and dense layers.
        # Use transformers.initialization wrappers (imported as `init`); they check the `_is_hf_initialized`
        # flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)


class DeepCpgDnaModel(DeepCpgDnaPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import DeepCpgDnaConfig, DeepCpgDnaModel, DnaTokenizer
        >>> config = DeepCpgDnaConfig()
        >>> model = DeepCpgDnaModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepcpgdna-smallwood2014-serum")
        >>> input = tokenizer(["ACGT" * 250 + "A", "TGCA" * 250 + "T"], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 128])
    """

    def __init__(self, config: DeepCpgDnaConfig):
        super().__init__(config)
        self.embeddings = DeepCpgDnaEmbedding(config)
        self.encoder = DeepCpgDnaEncoder(config)
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
    ) -> DeepCpgDnaModelOutput:
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
        # The DeepCpG-DNA encoder collapses the sequence dimension through its dense bottleneck, so the final
        # bottleneck embedding is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return DeepCpgDnaModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class DeepCpgDnaForSequencePrediction(DeepCpgDnaPreTrainedModel):
    """
    The per-cell methylation (final dense) layer of DeepCpG-DNA is **dataset-specific**: it has one output per single
    cell in the training dataset. `num_labels` therefore equals the number of cells in the chosen dataset (18 for the
    shipped Smallwood2014 serum mESC checkpoint) and is exposed through the shared
    [`SequencePredictionHead`][multimolecule.SequencePredictionHead] decoder.

    Examples:
        >>> import torch
        >>> from multimolecule import DeepCpgDnaConfig, DeepCpgDnaForSequencePrediction, DnaTokenizer
        >>> config = DeepCpgDnaConfig()
        >>> model = DeepCpgDnaForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepcpgdna-smallwood2014-serum")
        >>> input = tokenizer(["ACGT" * 250 + "A", "TGCA" * 250 + "T"], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2, config.num_labels)))
        >>> output["logits"].shape
        torch.Size([2, 18])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: DeepCpgDnaConfig):
        super().__init__(config)
        self.model = DeepCpgDnaModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"cell_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        return [f"cell_{index}" for index in range(self.config.num_labels)]

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


class DeepCpgDnaEmbedding(nn.Module):
    """One-hot embedding layer for DeepCpG-DNA.

    DeepCpG-DNA does not use learned word embeddings; it consumes a one-hot encoding of the four DNA nucleotides
    transposed into `(batch_size, vocab_size, sequence_length)` for the 1D convolution stack.
    """

    def __init__(self, config: DeepCpgDnaConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.sequence_length = config.sequence_length
        self.zero_token_id = config.vocab_size - 1
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
                raise ValueError("You have to specify input_ids when inputs_embeds is not provided")
            self._check_sequence_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            zero = input_ids == self.zero_token_id
            if invalid.any() or zero.any():
                inputs_embeds = inputs_embeds * (~(invalid | zero)).unsqueeze(-1).to(dtype)
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
                f"DeepCpG-DNA expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class DeepCpgDnaEncoder(nn.Module):
    def __init__(self, config: DeepCpgDnaConfig):
        super().__init__()
        layers = []
        in_channels = config.vocab_size
        for out_channels, kernel_size, pool_size in zip(
            config.conv_channels, config.conv_kernel_sizes, config.conv_pool_sizes
        ):
            layers.append(
                DeepCpgDnaConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                )
            )
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
        conv_output_size = self._conv_output_size(config)
        self.bottleneck = DeepCpgDnaBottleneck(config, conv_output_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.gradient_checkpointing = False

    @staticmethod
    def _conv_output_size(config: DeepCpgDnaConfig) -> int:
        # Upstream DeepCpG uses Keras Conv1D with `border_mode="valid"` and `MaxPooling1D` with default
        # `border_mode="valid"`. Valid conv trims by `kernel_size - 1`; valid max-pool with stride==pool_size
        # floor-divides.
        length = config.sequence_length
        for kernel_size, pool_size in zip(config.conv_kernel_sizes, config.conv_pool_sizes):
            length = length - kernel_size + 1
            length = length // pool_size
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
        # Keras `Flatten` reshapes the `(length, channels)` pooled feature map in row-major (length-major) order;
        # the bottleneck reconciles that ordering with PyTorch's `(channels, length)` layout.
        hidden_state = self.bottleneck(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class DeepCpgDnaConvLayer(nn.Module):
    """Convolution + activation + max-pool block.

    Upstream `CnnL2h128` applies Conv1D (valid padding) -> ReLU -> MaxPool1D (valid padding). The pool's default
    `stride` equals `pool_length`, so the sequence length is floor-divided at every block.
    """

    def __init__(
        self,
        config: DeepCpgDnaConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
    ):
        super().__init__()
        # Keras `border_mode="valid"` == torch `padding=0`.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.activation = ACT2FN[config.hidden_act]
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.pool(hidden_state)
        return hidden_state


class DeepCpgDnaBottleneck(nn.Module):
    """Dense bottleneck.

    Reproduces upstream `Flatten` + `Dense(128)`: a Keras channels-last (length-major) flatten followed by a bias-ful
    `Dense` linear projection to the bottleneck size.
    """

    def __init__(self, config: DeepCpgDnaConfig, conv_output_size: int):
        super().__init__()
        self.dense = nn.Linear(conv_output_size, config.bottleneck_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        # hidden_state is (batch, channels, length). Keras `Flatten` reshapes (length, channels) in row-major order,
        # i.e. length-major. Transpose to (batch, length, channels) before flattening so the MultiMolecule order
        # matches the upstream Dense kernel layout (which the converter consumes column-major over Keras's order).
        hidden_state = hidden_state.transpose(1, 2).flatten(1)
        return self.dense(hidden_state)


@dataclass
class DeepCpgDnaModelOutput(ModelOutput):
    """
    Base class for outputs of the DeepCpG-DNA backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Final bottleneck embedding produced by the DeepCpG-DNA encoder.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Same tensor as `last_hidden_state`; DeepCpG-DNA collapses the sequence dimension in its encoder.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple containing the one-hot embedding output and the final bottleneck embedding.
        attentions:
            Always `None`; DeepCpG-DNA is a convolutional model and has no attention layers. Provided for compatibility
            with the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


DeepCpgDnaPreTrainedModel._can_record_outputs = {"hidden_states": DeepCpgDnaEncoder}
