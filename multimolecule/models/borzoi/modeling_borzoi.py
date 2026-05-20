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
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import (
    HeadConfig,
    TokenPredictionHead,
    preserve_batch_norm_stats,
)

from ..modeling_outputs import TokenPredictorOutput
from .configuration_borzoi import BorzoiConfig


class BorzoiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BorzoiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["BorzoiLayer", "BorzoiConvLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, BorzoiAttention):
            init.normal_(module.rel_content_bias)
            init.normal_(module.rel_pos_bias)
            init.zeros_(module.to_out.weight)
            init.zeros_(module.to_out.bias)


class BorzoiModel(BorzoiPreTrainedModel):
    """
    The bare Borzoi backbone. Consumes a long DNA window and returns binned hidden states.

    The architecture follows the upstream Borzoi trunk: a pre-activation convolution stem with attention-pool
    downsampling, a width-growing residual convolution tower, a U-Net bottleneck pool, a Transformer trunk
    with Transformer-XL style relative positional encoding, two U-Net upsampling stages with depthwise-separable
    convolutions, and a center-crop. The positional axis of the output is *binned*: a window of
    `config.sequence_length` base pairs is downsampled and then re-upsampled, and `last_hidden_state` has shape
    `(batch_size, target_length, head_hidden_size)`.

    Examples:
        >>> from multimolecule import BorzoiConfig, BorzoiModel
        >>> config = BorzoiConfig(
        ...     sequence_length=512, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        ...     attention_head_size=4, attention_value_size=4, num_rel_pos_features=4,
        ...     stem_channels=8, conv_tower_channels=[12], head_hidden_size=8, target_length=16,
        ... )
        >>> model = BorzoiModel(config)
        >>> import torch
        >>> input_ids = torch.randint(config.vocab_size, (1, 512))
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 16, 8])
    """

    def __init__(self, config: BorzoiConfig):
        super().__init__(config)
        self.embeddings = BorzoiEmbedding(config)
        self.encoder = BorzoiEncoder(config)
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
    ) -> BaseModelOutput:
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
        encoder_outputs = self.encoder(embedding_output, **kwargs)

        return BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


class BorzoiForTokenPrediction(BorzoiPreTrainedModel):
    """
    Borzoi with a pointwise regression head over genomic coverage tracks.

    The binned positional axis is treated as the "token" axis: logits have shape
    `(batch_size, target_length, num_labels)` where `num_labels` is the number of coverage tracks
    of the configured `species` head.

    Examples:
        >>> import torch
        >>> from multimolecule import BorzoiConfig, BorzoiForTokenPrediction
        >>> config = BorzoiConfig(
        ...     sequence_length=512, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        ...     attention_head_size=4, attention_value_size=4, num_rel_pos_features=4,
        ...     stem_channels=8, conv_tower_channels=[12], head_hidden_size=8, target_length=16,
        ...     num_labels=4,
        ... )
        >>> model = BorzoiForTokenPrediction(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, 512))
        >>> output = model(input_ids, labels=torch.randn(1, 16, 4))
        >>> output["logits"].shape
        torch.Size([1, 16, 4])
    """

    def __init__(self, config: BorzoiConfig):
        super().__init__(config)
        self.model = BorzoiModel(config)
        token_head_config = HeadConfig(config.head) if config.head is not None else HeadConfig()
        if token_head_config.num_labels is None:
            token_head_config.num_labels = config.num_labels
        if token_head_config.hidden_size is None:
            token_head_config.hidden_size = config.head_hidden_size
        if token_head_config.problem_type is None:
            token_head_config.problem_type = "regression"
        self.token_head = TokenPredictionHead(config, token_head_config)
        self.head_config = self.token_head.config
        # Borzoi applies softplus to the per-track predictions so coverage stays non-negative.
        self.output_act = _resolve_activation(config.output_act)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [
                str(id2label.get(index, f"{self.config.species}_track_{index}"))
                for index in range(self.config.num_labels)
            ]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        return [f"{self.config.species}_track_{index}" for index in range(self.config.num_labels)]

    def postprocess(self, outputs: TokenPredictorOutput | ModelOutput | Tensor) -> tuple[Tensor, list[str]]:
        r"""Return the non-negative per-track coverage prediction with track channel names."""
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]
        coverage = self.output_act(logits) if self.output_act is not None else logits
        return coverage, self.output_channels

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        head_outputs = BaseModelOutput(last_hidden_state=outputs.last_hidden_state)

        # The binned axis has no special tokens; pass an all-ones mask so the shared head keeps every bin.
        bin_mask = outputs.last_hidden_state.new_ones(outputs.last_hidden_state.shape[:2], dtype=torch.long)
        output = self.token_head(head_outputs, bin_mask, None, None)
        logits = output.logits  # pre-activation; non-negative coverage = output_act(logits), exposed via postprocess

        loss = None
        if labels is not None:
            activated = self.output_act(logits) if self.output_act is not None else logits
            loss = self.token_head.criterion(activated, labels)

        return TokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BorzoiEmbedding(nn.Module):
    """One-hot feature projection following the MultiMolecule DNA token order."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16) so F.one_hot
        # output (always int64) can be cast to the active dtype in forward.
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
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            inputs_embeds = F.one_hot(
                input_ids.clamp(min=0, max=self.vocab_size - 1),
                num_classes=self.vocab_size,
            ).to(dtype=dtype)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class BorzoiEncoder(nn.Module):
    """Convolution stem + reducing tower + U-Net bottleneck + Transformer trunk + U-Net upsampling tail."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.config = config

        self.stem = BorzoiStem(config)

        tower: list[nn.Module] = []
        in_channels = config.stem_channels
        for index, out_channels in enumerate(config.conv_tower_channels):
            # The upstream tower applies a MaxPool *between* successive ConvBlocks; equivalently every conv stage
            # except the last is followed by a pool. The last stage is the skip target `x_unet0` and is *not*
            # pooled here -- the pool is the responsibility of the bottleneck `unet1` module below.
            apply_pool = index < len(config.conv_tower_channels) - 1
            tower.append(
                BorzoiConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.conv_tower_kernel_size,
                    pool_size=2 if apply_pool else 1,
                )
            )
            in_channels = out_channels
        self.conv_tower = nn.ModuleList(tower)
        self.skip0_channels = in_channels

        # U-Net bottleneck: pool + ConvBlock projecting to `hidden_size`. Output is the second skip `x_unet1`.
        self.unet_bottleneck = BorzoiConvLayer(
            config,
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.conv_tower_kernel_size,
            pool_size=2,
            pool_first=True,
        )
        self.skip1_channels = config.hidden_size

        # Final 2x pool before the Transformer trunk.
        self.trunk_pool = nn.MaxPool1d(2)

        # Horizontal projections of the two skip features into `hidden_size` channels.
        self.skip0_proj = BorzoiPointwiseBlock(config, in_channels=self.skip0_channels, out_channels=config.hidden_size)
        self.skip1_proj = BorzoiPointwiseBlock(config, in_channels=self.skip1_channels, out_channels=config.hidden_size)

        self.layers = nn.ModuleList([BorzoiLayer(config) for _ in range(config.num_hidden_layers)])

        # U-Net upsampling tail: two stages each doing pointwise conv + 2x upsample + skip-add + depthwise-separable
        # convolution at `unet_kernel_size`. Upstream uses two independent ConvBlocks (the `nn.Upsample` itself
        # is parameter-free and aliased between the two stages), so we instantiate two `BorzoiUpsampleBlock`s.
        self.upsample1 = BorzoiUpsampleBlock(config)
        self.upsample0 = BorzoiUpsampleBlock(config)
        self.separable1 = BorzoiSeparableConvBlock(
            config, channels=config.hidden_size, kernel_size=config.unet_kernel_size
        )
        self.separable0 = BorzoiSeparableConvBlock(
            config, channels=config.hidden_size, kernel_size=config.unet_kernel_size
        )

        self.crop = BorzoiTargetLengthCrop(config.target_length)

        self.head = BorzoiHead(config, dim_in=config.hidden_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        record_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states) or kwargs.get(
            "output_contexts", self.config.output_contexts
        )
        all_hidden_states: tuple[Tensor, ...] | None = () if record_hidden_states else None

        hidden_state = self.stem(hidden_state)
        # Reducing tower: every stage emits the wide-stream feature at the corresponding resolution.
        for layer in self.conv_tower:
            hidden_state = layer(hidden_state)
        x_unet0 = hidden_state  # `(B, skip0_channels, L / 2 ** (num_downsamples - 2))`

        x_unet1 = self.unet_bottleneck(x_unet0)  # `(B, hidden_size, L / 2 ** (num_downsamples - 1))`

        # Final pool to reach the Transformer resolution.
        hidden_state = self.trunk_pool(x_unet1)
        if record_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]

        # Project skip features into the Transformer channel count.
        skip0 = self.skip0_proj(x_unet0)
        skip1 = self.skip1_proj(x_unet1)

        # Transformer trunk on `(B, L_trunk, hidden_size)`.
        hidden_state = hidden_state.transpose(1, 2)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(layer.__call__, hidden_state)
            else:
                hidden_state = layer(hidden_state)
            if record_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)  # type: ignore[operator]
        hidden_state = hidden_state.transpose(1, 2)

        # U-Net upsampling tail: pointwise + upsample, skip-add, depthwise-separable conv. Each stage doubles
        # the positional axis until we are back to the resolution of `x_unet0`.
        if self.gradient_checkpointing and self.training:
            hidden_state = self._gradient_checkpointing_func(
                self.upsample1.__call__,
                hidden_state,
                context_fn=lambda module=self.upsample1: (
                    nullcontext(),
                    preserve_batch_norm_stats(module),
                ),
            )
            hidden_state = hidden_state + skip1
            hidden_state = self._gradient_checkpointing_func(
                self.separable1.__call__,
                hidden_state,
                context_fn=lambda module=self.separable1: (
                    nullcontext(),
                    preserve_batch_norm_stats(module),
                ),
            )
            hidden_state = self._gradient_checkpointing_func(
                self.upsample0.__call__,
                hidden_state,
                context_fn=lambda module=self.upsample0: (
                    nullcontext(),
                    preserve_batch_norm_stats(module),
                ),
            )
            hidden_state = hidden_state + skip0
            hidden_state = self._gradient_checkpointing_func(
                self.separable0.__call__,
                hidden_state,
                context_fn=lambda module=self.separable0: (
                    nullcontext(),
                    preserve_batch_norm_stats(module),
                ),
            )
        else:
            hidden_state = self.upsample1(hidden_state)
            hidden_state = hidden_state + skip1
            hidden_state = self.separable1(hidden_state)
            hidden_state = self.upsample0(hidden_state)
            hidden_state = hidden_state + skip0
            hidden_state = self.separable0(hidden_state)

        # `crop` operates on the (B, L, C) layout.
        hidden_state = self.crop(hidden_state.transpose(1, 2))
        last_hidden_state = self.head(hidden_state.transpose(1, 2)).transpose(1, 2)
        if record_hidden_states:
            all_hidden_states = all_hidden_states + (last_hidden_state,)  # type: ignore[operator]
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=all_hidden_states)


class BorzoiStem(nn.Module):
    """First convolution + 2x max-pool downsampling."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.vocab_size,
            config.stem_channels,
            kernel_size=config.stem_kernel_size,
            padding="same",
        )
        self.pool1 = nn.MaxPool1d(2)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.pool1(self.conv1(hidden_state))


class BorzoiLayer(nn.Module):
    """A Transformer block: relative-position attention sublayer + feed-forward sublayer."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.attention = BorzoiAttention(config)
        self.intermediate = BorzoiIntermediate(config)

    def forward(self, hidden_state: Tensor) -> Tensor:
        attention_output, _ = self.attention(hidden_state)
        hidden_state = hidden_state + attention_output
        hidden_state = hidden_state + self.intermediate(hidden_state)
        return hidden_state


class BorzoiAttention(nn.Module):
    """Multi-head attention with Transformer-XL style relative positional encoding."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.dim_key = config.attention_head_size
        self.dim_value = config.attention_value_size
        self.num_rel_pos_features = config.num_rel_pos_features
        self.scale = self.dim_key**-0.5

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
        self.to_q = nn.Linear(config.hidden_size, self.dim_key * self.num_heads, bias=False)
        self.to_k = nn.Linear(config.hidden_size, self.dim_key * self.num_heads, bias=False)
        self.to_v = nn.Linear(config.hidden_size, self.dim_value * self.num_heads, bias=False)
        self.to_rel_k = nn.Linear(self.num_rel_pos_features, self.dim_key * self.num_heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, self.num_heads, 1, self.dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, self.num_heads, 1, self.dim_key))
        self.to_out = nn.Linear(self.dim_value * self.num_heads, config.hidden_size)

        self.position_dropout = nn.Dropout(config.position_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.intermediate_dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, length, _ = tensor.shape
        return tensor.view(batch_size, length, self.num_heads, -1).permute(0, 2, 1, 3)

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        residual_input = self.layer_norm(hidden_state)
        length, device, dtype = (
            residual_input.shape[-2],
            residual_input.device,
            self.to_rel_k.weight.dtype,
        )

        query = self._split_heads(self.to_q(residual_input)) * self.scale
        key = self._split_heads(self.to_k(residual_input))
        value = self._split_heads(self.to_v(residual_input))

        content_logits = torch.einsum("b h i d, b h j d -> b h i j", query + self.rel_content_bias, key)

        positions = _get_positional_embed(length, self.num_rel_pos_features, device, dtype)
        positions = self.position_dropout(positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rel_k.view(rel_k.shape[0], self.num_heads, -1).permute(1, 0, 2)
        rel_logits = torch.einsum("b h i d, h j d -> b h i j", query + self.rel_pos_bias, rel_k)
        rel_logits = _relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        context = torch.einsum("b h i j, b h j d -> b h i d", attn, value)
        context = context.permute(0, 2, 1, 3).reshape(context.shape[0], length, -1)
        return self.dropout(self.to_out(context)), attn


class BorzoiIntermediate(nn.Module):
    """Feed-forward sublayer of a Transformer block."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
        self.dense1 = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.dropout1 = nn.Dropout(config.intermediate_dropout)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout2 = nn.Dropout(config.intermediate_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.layer_norm(hidden_state)
        hidden_state = self.dense1(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dense2(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        return hidden_state


class BorzoiConvLayer(nn.Module):
    """
    Borzoi pre-activation convolution block (`conv_nac` family).

    Order: `BatchNorm -> activation -> Conv1d -> [MaxPool]`. When `pool_first=True`, the optional pool runs
    *before* the conv block instead of after; this matches the U-Net bottleneck stage in the upstream graph.
    """

    def __init__(
        self,
        config: BorzoiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 1,
        pool_first: bool = False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.pre_pool = nn.MaxPool1d(pool_size) if (pool_first and pool_size > 1) else None
        self.batch_norm1 = nn.BatchNorm1d(in_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.pool1 = nn.MaxPool1d(pool_size) if (not pool_first and pool_size > 1) else None

    def forward(self, hidden_state: Tensor) -> Tensor:
        if self.pre_pool is not None:
            hidden_state = self.pre_pool(hidden_state)
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        if self.pool1 is not None:
            hidden_state = self.pool1(hidden_state)
        return hidden_state


class BorzoiPointwiseBlock(nn.Module):
    """Pre-activation pointwise (`kernel_size=1`) convolution block used by the U-Net skip projections."""

    def __init__(self, config: BorzoiConfig, in_channels: int, out_channels: int):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(in_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        return self.conv1(hidden_state)


class BorzoiUpsampleBlock(nn.Module):
    """Pointwise pre-activation conv followed by 2x nearest-neighbour upsampling."""

    def __init__(self, config: BorzoiConfig):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        return self.upsample(hidden_state)


class BorzoiSeparableConvBlock(nn.Module):
    """
    Depthwise-separable convolution block used by the U-Net upsampling tail.

    Mirrors the upstream `conv_type="separable"` branch: depthwise convolution producing `channels` outputs
    followed by a pointwise convolution. The upstream graph applies no batch norm / activation in the separable
    branch, matching the `nn.Identity` / `nn.Identity` placeholders in the reference implementation.
    """

    def __init__(self, config: BorzoiConfig, channels: int, kernel_size: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding="same",
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.depthwise(hidden_state)
        hidden_state = self.pointwise(hidden_state)
        return hidden_state


class BorzoiTargetLengthCrop(nn.Module):
    """Center-crop the binned tail output down to `target_length` positions."""

    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, hidden_state: Tensor) -> Tensor:
        length, target = hidden_state.shape[-2], self.target_length
        if target == -1 or length == target:
            return hidden_state
        if length < target:
            raise ValueError(f"sequence length {length} is less than target length {target}")
        trim = (length - target) // 2
        return hidden_state[:, trim : trim + target]


class BorzoiHead(nn.Module):
    """Pointwise output head: pre-activation conv expanding to `head_hidden_size`, dropout, and final activation."""

    def __init__(self, config: BorzoiConfig, dim_in: int):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(dim_in, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = ACT2FN[config.hidden_act]  # pre-activation before the pointwise projection
        self.conv1 = nn.Conv1d(dim_in, config.head_hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.act2 = ACT2FN[config.hidden_act]  # post-projection activation before the track head

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return self.act2(hidden_state)


def _resolve_activation(name: str | None):
    if name is None:
        return None
    if name == "softplus":
        return nn.Softplus()
    return ACT2FN[name]


def _get_positional_features_central_mask(positions: Tensor, features: int, seq_len: int, dtype=torch.float) -> Tensor:
    # Upstream Borzoi uses the geometric central-mask basis (`pow_rate = (seq_len + 1) ** (1 / features)`),
    # which differs from Enformer's `2 ** k` central mask.
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(
        torch.tensor(pow_rate, device=positions.device, dtype=dtype),
        torch.arange(1, features + 1, device=positions.device, dtype=dtype),
    )
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None].to(dtype)).to(dtype)


def _get_positional_embed(seq_len: int, feature_size: int, device, dtype=torch.float) -> Tensor:
    # Borzoi uses a single basis family (central mask) with a signed copy, i.e. 2 components.
    num_components = 2
    if (feature_size % num_components) != 0:
        raise ValueError(f"feature size is not divisible by number of components ({num_components})")
    num_basis_per_class = feature_size // num_components
    distances = torch.arange(-seq_len + 1, seq_len, device=device)
    embeddings = _get_positional_features_central_mask(distances, num_basis_per_class, seq_len, dtype=dtype)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None].to(dtype) * embeddings), dim=-1)
    return embeddings.to(dtype)


def _relative_shift(x: Tensor) -> Tensor:
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# Expose per-layer attention weights for interpretability via the transformers output-capture mechanism.
# `BorzoiAttention.forward` returns `(output, attn)`, so the recorder captures the attention tensor at tuple
# index 1. Recording is opt-in (`output_attentions=True` / `config.output_attentions`).
BorzoiPreTrainedModel._can_record_outputs = {
    "attentions": OutputRecorder(BorzoiAttention, index=1),
}
