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


def _dense_bias_from_score_mod(
    score_mod: Callable, batch_size: int, num_heads: int, seq_len: int, *, device, dtype
) -> Tensor:
    r"""
    Materialize a ``score_mod`` into a dense additive bias for the SDPA/eager path.
    """
    batch = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1)
    head = torch.arange(num_heads, device=device).view(1, num_heads, 1, 1)
    q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
    base = torch.zeros((1, num_heads, seq_len, seq_len), device=device, dtype=dtype)
    return score_mod(base, batch, head, q_idx, kv_idx)


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
    is_causal: bool = False,
    score_mod: Callable | None = None,
    alibi_slopes: Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[Tensor, Tensor | None]:
    # ALiBi can't ride the packed flash path: torch's bundled FlashAttention rejects alibi_slopes,
    # so it falls back to a FlexAttention score_mod. A real flash+ALiBi win would need the external
    # flash_attn library's varlen kernel.
    if alibi_slopes is not None and score_mod is None:
        score_mod = _alibi_score_mod(alibi_slopes)
    if score_mod is not None and isinstance(query, NestedTensor):
        # flex_attention returns (batch, heads, seq, dim); interfaces return (batch, seq, heads, dim).
        return flex_attn.flex_attention(query, key, value, score_mod=score_mod, scale=scaling).transpose(1, 2), None
    if score_mod is not None:
        bias = _dense_bias_from_score_mod(
            score_mod, query.shape[0], query.shape[1], query.shape[-2], device=query.device, dtype=query.dtype
        )
        if attention_mask is None:
            attention_mask = bias
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=bias.dtype).masked_fill(
                    ~attention_mask, float("-inf")
                )
            attention_mask = attention_mask + bias
    if isinstance(query, NestedTensor) and attention_mask is not None:
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
