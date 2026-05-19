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

from multimolecule.modules import HeadConfig, TokenPredictionHead

from ..modeling_outputs import TokenPredictorOutput
from .configuration_enformer import EnformerConfig


class EnformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EnformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["EnformerLayer", "EnformerConvLayer"]

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
        elif isinstance(module, EnformerAttention):
            init.normal_(module.rel_content_bias)
            init.normal_(module.rel_pos_bias)
            init.zeros_(module.to_out.weight)
            init.zeros_(module.to_out.bias)
        elif isinstance(module, EnformerAttentionPool):
            # `to_attn_logits` is a 1x1 Conv2d whose weight is a persistent parameter, so it is
            # restored from the checkpoint on `from_pretrained`. The `init.dirac_` wrapper
            # respects `_is_hf_initialized` (no-op for loaded weights); we then scale by 2 to
            # reproduce the upstream average-pooling initialisation only when the weight was
            # actually (re)initialised here.
            was_initialized = getattr(module.to_attn_logits.weight, "_is_hf_initialized", False)
            init.dirac_(module.to_attn_logits.weight)
            if not was_initialized:
                with torch.no_grad():
                    module.to_attn_logits.weight.mul_(2)


class EnformerModel(EnformerPreTrainedModel):
    """
    The bare Enformer backbone. Consumes a long DNA window and returns binned hidden states.

    The positional axis of the output is *binned*: a window of `config.sequence_length` base pairs
    is downsampled by the convolution stem, processed by the Transformer trunk, and center-cropped
    so `last_hidden_state` has shape `(batch_size, target_length, head_hidden_size)`.

    Examples:
        >>> from multimolecule import EnformerConfig, EnformerModel
        >>> config = EnformerConfig(
        ...     sequence_length=256, hidden_size=12, num_hidden_layers=1, num_attention_heads=2,
        ...     attention_head_size=4, num_downsamples=3, dim_divisible_by=2, target_length=16,
        ... )
        >>> model = EnformerModel(config)
        >>> import torch
        >>> input_ids = torch.randint(config.vocab_size, (1, 256))
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 16, 24])
    """

    def __init__(self, config: EnformerConfig):
        super().__init__(config)
        self.embeddings = EnformerEmbedding(config)
        self.encoder = EnformerEncoder(config)
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


class EnformerForTokenPrediction(EnformerPreTrainedModel):
    """
    Enformer with a pointwise regression head over genomic coverage tracks.

    The binned positional axis is treated as the "token" axis: logits have shape
    `(batch_size, target_length, num_labels)` where `num_labels` is the number of coverage tracks
    of the configured `species` head.

    Examples:
        >>> import torch
        >>> from multimolecule import EnformerConfig, EnformerForTokenPrediction
        >>> config = EnformerConfig(
        ...     sequence_length=256, hidden_size=12, num_hidden_layers=1, num_attention_heads=2,
        ...     attention_head_size=4, num_downsamples=3, dim_divisible_by=2, target_length=16,
        ...     num_labels=4,
        ... )
        >>> model = EnformerForTokenPrediction(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, 256))
        >>> output = model(input_ids, labels=torch.randn(1, 16, 4))
        >>> output["logits"].shape
        torch.Size([1, 16, 4])
    """

    def __init__(self, config: EnformerConfig):
        super().__init__(config)
        self.model = EnformerModel(config)
        token_head_config = HeadConfig(config.head) if config.head is not None else HeadConfig()
        if token_head_config.num_labels is None:
            token_head_config.num_labels = config.num_labels
        if token_head_config.hidden_size is None:
            token_head_config.hidden_size = config.head_hidden_size
        if token_head_config.problem_type is None:
            token_head_config.problem_type = "regression"
        self.token_head = TokenPredictionHead(config, token_head_config)
        self.head_config = self.token_head.config
        # Enformer applies softplus to the per-track predictions so coverage stays non-negative.
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

        # The binned axis has no special tokens; pass an all-ones mask so the shared head keeps
        # every bin.
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


class EnformerEmbedding(nn.Module):
    """One-hot feature projection following the MultiMolecule DNA token order."""

    def __init__(self, config: EnformerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
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
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), num_classes=self.vocab_size).to(
                dtype=dtype
            )
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class EnformerEncoder(nn.Module):
    """Convolution stem + conv tower followed by the Transformer trunk and pointwise head."""

    def __init__(self, config: EnformerConfig):
        super().__init__()
        self.config = config
        half_dim = config.hidden_size // 2

        self.stem = EnformerStem(config)

        filter_list = _exponential_linspace_int(
            half_dim, config.hidden_size, num=config.num_downsamples - 1, divisible_by=config.dim_divisible_by
        )
        filter_list = [half_dim, *filter_list]
        self.conv_tower = nn.ModuleList(
            [EnformerConvLayer(config, dim_in, dim_out) for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:])]
        )

        self.layers = nn.ModuleList([EnformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.crop = EnformerTargetLengthCrop(config.target_length)
        self.head = EnformerHead(config, filter_list[-1])
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
        for conv_layer in self.conv_tower:
            hidden_state = conv_layer(hidden_state)
        if record_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]

        hidden_state = hidden_state.transpose(1, 2)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(layer.__call__, hidden_state)
            else:
                hidden_state = layer(hidden_state)
            if record_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)  # type: ignore[operator]

        hidden_state = self.crop(hidden_state)
        last_hidden_state = self.head(hidden_state)
        if record_hidden_states:
            all_hidden_states = all_hidden_states + (last_hidden_state,)  # type: ignore[operator]
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=all_hidden_states)


class EnformerStem(nn.Module):
    """First convolution, a residual convolution block, and attention pooling."""

    def __init__(self, config: EnformerConfig):
        super().__init__()
        half_dim = config.hidden_size // 2
        self.conv1 = nn.Conv1d(
            config.vocab_size,
            half_dim,
            kernel_size=config.stem_kernel_size,
            padding=config.stem_kernel_size // 2,
        )
        self.conv_block = EnformerConvBlock(config, half_dim, half_dim, kernel_size=1)
        self.pool = EnformerAttentionPool(half_dim)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv1(hidden_state)
        hidden_state = hidden_state + self.conv_block(hidden_state)
        return self.pool(hidden_state)


class EnformerConvLayer(nn.Module):
    """A conv-tower stage: a strided conv block, a residual pointwise block, and attention pooling."""

    def __init__(self, config: EnformerConfig, dim_in: int, dim_out: int):
        super().__init__()
        self.conv_block = EnformerConvBlock(config, dim_in, dim_out, kernel_size=config.conv_tower_kernel_size)
        self.conv_block_residual = EnformerConvBlock(config, dim_out, dim_out, kernel_size=1)
        self.pool = EnformerAttentionPool(dim_out)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv_block(hidden_state)
        hidden_state = hidden_state + self.conv_block_residual(hidden_state)
        return self.pool(hidden_state)


class EnformerConvBlock(nn.Module):
    """BatchNorm -> activation -> convolution, the basic conv unit used throughout the stem."""

    def __init__(self, config: EnformerConfig, dim_in: int, dim_out: int, kernel_size: int = 1):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(dim_in, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        return self.conv1(hidden_state)


class EnformerAttentionPool(nn.Module):
    """Learned softmax pooling that downsamples the sequence axis by `pool_size`."""

    def __init__(self, dim: int, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, hidden_state: Tensor) -> Tensor:
        batch_size, _, length = hidden_state.shape
        remainder = length % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            padding = self.pool_size - remainder
            hidden_state = F.pad(hidden_state, (0, padding), value=0)
            mask = torch.zeros((batch_size, 1, length), dtype=torch.bool, device=hidden_state.device)
            mask = F.pad(mask, (0, padding), value=True)

        hidden_state = hidden_state.reshape(batch_size, hidden_state.shape[1], -1, self.pool_size)
        logits = self.to_attn_logits(hidden_state)

        if needs_padding:
            mask = mask.reshape(batch_size, 1, -1, self.pool_size)
            logits = logits.masked_fill(mask, -torch.finfo(logits.dtype).max)

        attn = logits.softmax(dim=-1)
        return (hidden_state * attn).sum(dim=-1)


class EnformerTargetLengthCrop(nn.Module):
    """Center-crop the binned trunk output down to `target_length` positions."""

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


class EnformerLayer(nn.Module):
    """A Transformer block: a relative-position attention sublayer and a feed-forward sublayer."""

    def __init__(self, config: EnformerConfig):
        super().__init__()
        self.attention = EnformerAttention(config)
        self.intermediate = EnformerIntermediate(config)

    def forward(self, hidden_state: Tensor) -> Tensor:
        attention_output, _ = self.attention(hidden_state)
        hidden_state = hidden_state + attention_output
        hidden_state = hidden_state + self.intermediate(hidden_state)
        return hidden_state


class EnformerAttention(nn.Module):
    """Multi-head attention with Transformer-XL style relative positional encoding."""

    def __init__(self, config: EnformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.dim_key = config.attention_head_size
        self.dim_value = config.hidden_size // config.num_attention_heads
        self.num_rel_pos_features = config.hidden_size // config.num_attention_heads
        self.scale = self.dim_key**-0.5

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.to_q = nn.Linear(config.hidden_size, self.dim_key * self.num_heads, bias=False)
        self.to_k = nn.Linear(config.hidden_size, self.dim_key * self.num_heads, bias=False)
        self.to_v = nn.Linear(config.hidden_size, self.dim_value * self.num_heads, bias=False)
        self.to_rel_k = nn.Linear(self.num_rel_pos_features, self.dim_key * self.num_heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, self.num_heads, 1, self.dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, self.num_heads, 1, self.dim_key))
        self.to_out = nn.Linear(self.dim_value * self.num_heads, config.hidden_size)

        self.position_dropout = nn.Dropout(config.position_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.use_precomputed_gamma_basis = config.use_precomputed_gamma_basis
        if config.use_precomputed_gamma_basis:
            gamma_basis_shape = (2 * config.num_bins - 1, self.num_rel_pos_features // 6)
            gamma_position_basis = torch.full(gamma_basis_shape, float("nan"))
        else:
            gamma_position_basis = torch.empty(0)
        self.register_buffer(
            "gamma_position_basis",
            gamma_position_basis,
            persistent=config.use_precomputed_gamma_basis,
        )

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, length, _ = tensor.shape
        return tensor.view(batch_size, length, self.num_heads, -1).permute(0, 2, 1, 3)

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        residual_input = self.layer_norm(hidden_state)
        length, device, dtype = residual_input.shape[-2], residual_input.device, self.to_rel_k.weight.dtype

        query = self._split_heads(self.to_q(residual_input)) * self.scale
        key = self._split_heads(self.to_k(residual_input))
        value = self._split_heads(self.to_v(residual_input))

        content_logits = torch.einsum("b h i d, b h j d -> b h i j", query + self.rel_content_bias, key)

        positions = _get_positional_embed(
            length,
            self.num_rel_pos_features,
            device,
            dtype,
            use_precomputed_gamma_basis=self.use_precomputed_gamma_basis,
            gamma_position_basis=self.gamma_position_basis,
        )
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


class EnformerIntermediate(nn.Module):
    """Feed-forward sublayer of a Transformer block."""

    def __init__(self, config: EnformerConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dense1 = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout2 = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.layer_norm(hidden_state)
        hidden_state = self.dense1(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dense2(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        return hidden_state


class EnformerHead(nn.Module):
    """Pointwise output head: a conv block expanding to `head_hidden_size`, then activation."""

    def __init__(self, config: EnformerConfig, dim_in: int):
        super().__init__()
        self.conv_block = EnformerConvBlock(config, dim_in, config.head_hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(config.hidden_dropout / 8)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = self.conv_block(hidden_state)
        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = self.dropout(hidden_state)
        return self.act(hidden_state)


def _resolve_activation(name: str | None):
    if name is None:
        return None
    if name == "softplus":
        return nn.Softplus()
    return ACT2FN[name]


def _exponential_linspace_int(start: int, end: int, num: int, divisible_by: int = 1) -> list[int]:
    def _round(x: float) -> int:
        return int(round(x / divisible_by) * divisible_by)

    if num < 1:
        return []
    if num == 1:
        return [_round(end)]
    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def _get_positional_features_exponential(
    positions: Tensor, features: int, seq_len: int, min_half_life: float = 3.0, dtype=torch.float
) -> Tensor:
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def _get_positional_features_central_mask(positions: Tensor, features: int, seq_len: int, dtype=torch.float) -> Tensor:
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)


def _gamma_pdf(x: Tensor, concentration: Tensor, rate: Tensor) -> Tensor:
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def _get_positional_features_gamma(
    positions: Tensor, features: int, seq_len: int, stddev=None, start_mean=None, eps: float = 1e-8, dtype=torch.float
) -> Tensor:
    if stddev is None:
        stddev = seq_len / (2 * features)
    if start_mean is None:
        start_mean = seq_len / features
    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = _gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    return probabilities / torch.amax(probabilities, dim=-1, keepdim=True)


def _get_precomputed_gamma_basis(
    gamma_position_basis: Tensor | None,
    seq_len: int,
    features: int,
    device,
    dtype=torch.float,
) -> Tensor:
    if seq_len != 1536:
        raise ValueError("Precomputed Enformer gamma basis is only defined for sequence length 196608")
    expected_shape = (2 * seq_len - 1, features)
    if gamma_position_basis is None or gamma_position_basis.numel() == 0:
        raise ValueError(
            "Enformer use_precomputed_gamma_basis=True requires converted gamma_position_basis buffers in the "
            "checkpoint"
        )
    if tuple(gamma_position_basis.shape) != expected_shape:
        raise ValueError(f"gamma_position_basis shape {tuple(gamma_position_basis.shape)} != {expected_shape}")
    if torch.isnan(gamma_position_basis).any():
        raise ValueError(
            "Enformer use_precomputed_gamma_basis=True requires loaded gamma_position_basis values, not an "
            "uninitialized buffer"
        )
    return gamma_position_basis.to(device=device, dtype=dtype)


def _get_positional_embed(
    seq_len: int,
    feature_size: int,
    device,
    dtype=torch.float,
    *,
    use_precomputed_gamma_basis: bool = False,
    gamma_position_basis: Tensor | None = None,
) -> Tensor:
    distances = torch.arange(-seq_len + 1, seq_len, device=device)
    num_components = 6
    if (feature_size % num_components) != 0:
        raise ValueError(f"feature size is not divisible by number of components ({num_components})")
    num_basis_per_class = feature_size // num_components
    gamma_features = (
        _get_precomputed_gamma_basis(gamma_position_basis, seq_len, num_basis_per_class, device, dtype)
        if use_precomputed_gamma_basis
        else _get_positional_features_gamma(distances, num_basis_per_class, seq_len, dtype=dtype)
    )
    features = [
        _get_positional_features_exponential(distances, num_basis_per_class, seq_len, dtype=dtype),
        _get_positional_features_central_mask(distances, num_basis_per_class, seq_len, dtype=dtype),
        gamma_features,
    ]
    embeddings = torch.cat(features, dim=-1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings.to(dtype)


def _relative_shift(x: Tensor) -> Tensor:
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# Expose per-layer attention weights for interpretability via the transformers
# output-capture mechanism. `EnformerAttention.forward` returns `(output, attn)`,
# so the recorder captures the attention tensor at tuple index 1. Recording is
# opt-in (`output_attentions=True` / `config.output_attentions`), so the default
# forward computation is unchanged.
EnformerPreTrainedModel._can_record_outputs = {
    "attentions": OutputRecorder(EnformerAttention, index=1),
}
