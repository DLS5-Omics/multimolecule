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

from collections.abc import Callable

import torch
import torch.nn.attention.flex_attention as flex_attn
from danling import NestedTensor
from torch import Tensor, nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .functional import eager_attention_forward


def attention_forward_function(attn_implementation: str | None) -> Callable:
    if attn_implementation in (None, "eager"):
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[attn_implementation]


def _dense_bias_from_score_mod(score_mod: Callable, query: Tensor, key: Tensor) -> Tensor:
    r"""
    Materialize a ``score_mod`` into a dense additive bias of shape ``(1, heads, query, key)``.

    Query positions are aligned to the last keys, so cached decoding sees the same bias as a full forward.
    """
    batch_size, num_heads, q_len, kv_len = query.shape[0], query.shape[1], query.shape[-2], key.shape[-2]
    device = query.device
    batch = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1)
    head = torch.arange(num_heads, device=device).view(1, num_heads, 1, 1)
    q_idx = torch.arange(kv_len - q_len, kv_len, device=device).view(1, 1, q_len, 1)
    kv_idx = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
    base = torch.zeros((1, num_heads, q_len, kv_len), device=device, dtype=query.dtype)
    return score_mod(base, batch, head, q_idx, kv_idx)


def _causal_mask_mod(batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    return q_idx >= kv_idx


def _nested_causal_block_mask(query: NestedTensor, key: NestedTensor):
    from danling.tensors import create_flex_block_mask

    return create_flex_block_mask(_causal_mask_mod, query, key)


def _alibi_score_mod(slopes: Tensor) -> Callable:
    r"""
    ALiBi as a FlexAttention ``score_mod`` (``score - slope[head] * |q - kv|``): the eager/flex
    equivalent of the ``alibi_slopes`` the flash kernel applies natively, so the non-flash paths
    derive the same bias from the same slopes.
    """

    def score_mod(score: Tensor, batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return score - slopes[head] * (q_idx - kv_idx).abs()

    return score_mod


def attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    *,
    attn_implementation: str | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool | None = None,
    score_mod: Callable | None = None,
    alibi_slopes: Tensor | None = None,
    attention_bias: Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[Tensor, Tensor | None]:
    r"""
    Dispatch attention to the configured backend.

    Args:
        attention_mask: Mask in the format transformers' mask utilities build for ``attn_implementation``.
        attention_bias: Additive score bias broadcastable to ``(batch, heads, query, key)``, such as relative
            position, ALiBi or pairwise biases. It is passed to the backend as ``position_bias`` instead of being
            folded into ``attention_mask``, so each backend combines it with its own mask format.
        is_causal: Whether attention is causal. Defaults to ``module.is_causal``.
    """
    # Causal mask builders return None when the backend can apply `is_causal` itself, so an unset flag must follow
    # the module rather than default to False.
    if is_causal is None:
        is_causal = getattr(module, "is_causal", False)
    # ALiBi can't ride the packed flash path: torch's bundled FlashAttention rejects alibi_slopes,
    # so it falls back to a FlexAttention score_mod. A real flash+ALiBi win would need the external
    # flash_attn library's varlen kernel.
    if alibi_slopes is not None and score_mod is None:
        score_mod = _alibi_score_mod(alibi_slopes)
    causal_flex = is_causal and attention_mask is None and attn_implementation == "flex_attention"
    if isinstance(query, NestedTensor) and attention_bias is None and (score_mod is not None or causal_flex):
        # Ragged inputs carry no materialized mask, so causality has to ride in a block mask.
        block_mask = _nested_causal_block_mask(query, key) if is_causal else None
        output = flex_attn.flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask, scale=scaling)
        # flex_attention returns (batch, heads, seq, dim); interfaces return (batch, seq, heads, dim).
        return output.transpose(1, 2), None
    if score_mod is not None:
        bias = _dense_bias_from_score_mod(score_mod, query, key)
        attention_bias = bias if attention_bias is None else attention_bias + bias
    if (
        attn_implementation == "flex_attention"
        and isinstance(attention_mask, Tensor)
        and not isinstance(query, NestedTensor)
    ):
        # transformers adds tensor masks to flex scores as `score_mask[b][0][q][kv]`, which torch 2.15 miscompiles
        # on CPU. An additive mask is just another bias, so fold it in; the mask then also covers causality.
        mask = attention_mask[..., : key.shape[-2]]
        attention_bias = mask if attention_bias is None else attention_bias + mask
        attention_mask, is_causal = None, False
    if attention_bias is not None:
        if "flash" in (attn_implementation or ""):
            # Flash kernels take no score bias, and transformers' flash path would drop it silently.
            raise ValueError(f"{attn_implementation} does not support relative position, ALiBi or pairwise biases.")
        attention_bias = attention_bias.to(query.dtype)
        if attn_implementation == "flex_attention":
            # transformers' flex reads `position_bias[b, h, q, kv]`, so the bias needs every dimension.
            attention_bias = attention_bias.expand(query.shape[0], query.shape[1], query.shape[-2], key.shape[-2])
        kwargs["position_bias"] = attention_bias
    if isinstance(query, NestedTensor) and (attention_mask is not None or attention_bias is not None):
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, is_causal=is_causal, **kwargs
        )
    attention_interface = attention_forward_function(attn_implementation)
    return attention_interface(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )
