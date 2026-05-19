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

from multimolecule.modules import SequencePredictionHead, preserve_batch_norm_stats

from ..modeling_outputs import SequencePredictorOutput
from .configuration_scbasset import ScBassetConfig


class ScBassetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ScBassetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["ScBassetConvLayer"]

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


class ScBassetModel(ScBassetPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import ScBassetConfig, ScBassetModel, DnaTokenizer
        >>> config = ScBassetConfig()
        >>> model = ScBassetModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/scbasset")
        >>> input = tokenizer(["ACGT" * 336, "TGCA" * 336], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 32])
    """

    def __init__(self, config: ScBassetConfig):
        super().__init__(config)
        self.embeddings = ScBassetEmbedding(config)
        self.encoder = ScBassetEncoder(config)
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
        # The scBasset encoder collapses the sequence dimension through its dense bottleneck, so the final
        # bottleneck embedding is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class ScBassetForSequencePrediction(ScBassetPreTrainedModel):
    """
    The cell-embedding (final dense) layer of scBasset is **dataset-specific**: it has one row per single cell in
    the training atlas. `num_labels` therefore equals the number of cells in the chosen dataset (2034 for the
    shipped Buenrostro2018 hematopoiesis checkpoint) and is exposed through the shared
    [`SequencePredictionHead`][multimolecule.SequencePredictionHead] decoder.

    Examples:
        >>> import torch
        >>> from multimolecule import ScBassetConfig, ScBassetForSequencePrediction, DnaTokenizer
        >>> config = ScBassetConfig()
        >>> model = ScBassetForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/scbasset")
        >>> input = tokenizer(["ACGT" * 336, "TGCA" * 336], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2, 2034)))
        >>> output["logits"].shape
        torch.Size([2, 2034])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: ScBassetConfig):
        super().__init__(config)
        self.model = ScBassetModel(config)
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


class ScBassetEmbedding(nn.Module):
    """One-hot embedding layer for scBasset.

    scBasset does not use learned word embeddings; it consumes a one-hot encoding of the four DNA nucleotides
    transposed into `(batch_size, vocab_size, sequence_length)` for the 1D convolution stack.
    """

    def __init__(self, config: ScBassetConfig):
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
                f"scBasset expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class ScBassetEncoder(nn.Module):
    def __init__(self, config: ScBassetConfig):
        super().__init__()
        layers = []
        in_channels = config.vocab_size
        layers.append(
            ScBassetConvLayer(
                config,
                in_channels=in_channels,
                out_channels=config.stem_channels,
                kernel_size=config.stem_kernel_size,
                pool_size=config.stem_pool_size,
            )
        )
        in_channels = config.stem_channels
        for out_channels in config.tower_channels:
            layers.append(
                ScBassetConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.tower_kernel_size,
                    pool_size=config.tower_pool_size,
                )
            )
            in_channels = out_channels
        # Final pointwise convolution (kernel size 1, no pooling).
        layers.append(
            ScBassetConvLayer(
                config,
                in_channels=in_channels,
                out_channels=config.pointwise_channels,
                kernel_size=1,
                pool_size=1,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.flattened_size = self._flattened_size(config)
        self.bottleneck = ScBassetBottleneck(config, self.flattened_size)
        self.activation = ACT2FN[config.hidden_act]
        self.gradient_checkpointing = False

    @staticmethod
    def _flattened_size(config: ScBassetConfig) -> int:
        # Keras MaxPool1D(padding="same") is ceil-mode pooling.
        length = -(-config.sequence_length // config.stem_pool_size)
        for _ in config.tower_channels:
            length = -(-length // config.tower_pool_size)
        return length * config.pointwise_channels

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
        # Upstream dense_block applies its activation BEFORE flattening; the flatten then uses Keras
        # channels-last (length-major) row-major order, reconciled by ScBassetBottleneck.
        hidden_state = self.activation(hidden_state)
        hidden_state = self.bottleneck(hidden_state)
        # scBasset applies one more GELU on the bottleneck embedding before the cell-embedding (head) layer.
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ScBassetConvLayer(nn.Module):
    """Pre-activation convolution block.

    scBasset's `conv_block` applies activation -> Conv1d(no bias) -> BatchNorm -> MaxPool. The activation comes
    *before* the convolution, so the very first block applies GELU to the one-hot input.
    """

    def __init__(
        self,
        config: ScBassetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
    ):
        super().__init__()
        self.activation = ACT2FN[config.hidden_act]
        # Keras Conv1D padding="same" with stride 1 == torch padding=kernel_size // 2 for odd kernels.
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.pool_size = pool_size

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        if self.pool_size > 1:
            hidden_state = _same_max_pool1d(hidden_state, self.pool_size)
        return hidden_state


class ScBassetBottleneck(nn.Module):
    """Dense bottleneck.

    Reproduces upstream `dense_block(flatten=True, batch_norm=True)`: a Keras channels-last (length-major)
    flatten, a bias-free `Dense`, and batch normalization, followed by dropout.
    """

    def __init__(self, config: ScBassetConfig, flattened_size: int):
        super().__init__()
        self.dense = nn.Linear(flattened_size, config.bottleneck_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(config.bottleneck_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        # hidden_state is (batch, channels, length). Keras Reshape flattens the (length, channels) tensor in
        # row-major order, i.e. length-major. Transpose to (batch, length, channels) before flattening so the
        # MultiMolecule order matches the upstream Dense kernel layout.
        hidden_state = hidden_state.transpose(1, 2).flatten(1)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


def _same_max_pool1d(hidden_state: Tensor, pool_size: int) -> Tensor:
    """Keras `MaxPool1D(pool_size, padding="same")`.

    TensorFlow "same" pooling produces `ceil(L / pool_size)` outputs and pads the right edge so the final
    window is full; the pad value does not affect the max because TF excludes padded positions. We replicate
    that by padding the right edge with `-inf` and using ceil-mode would double-count, so pad explicitly.
    """
    length = hidden_state.shape[-1]
    output_length = -(-length // pool_size)
    pad = output_length * pool_size - length
    if pad > 0:
        hidden_state = F.pad(hidden_state, (0, pad), value=float("-inf"))
    return F.max_pool1d(hidden_state, kernel_size=pool_size, stride=pool_size)
