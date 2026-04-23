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

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from ..outputs import AttentionOutput

AttentionAggregation = Literal["mean_heads", "mean_layers", "rollout"]


def prepare_attention_output(
    attentions: tuple[Tensor, ...] | list[Tensor] | Tensor | None,
    *,
    layers: int | Sequence[int] | None = None,
    heads: int | Sequence[int] | None = None,
    aggregate: AttentionAggregation | None = None,
    tokens: Sequence[str] | None = None,
) -> AttentionOutput:
    attention_tensor = stack_attentions(attentions)
    attention_tensor, selected_layers, selected_heads = select_attentions(
        attention_tensor,
        layers=layers,
        heads=heads,
    )
    attention_tensor = aggregate_attentions(attention_tensor, aggregate)
    output_layers, output_heads = _resolve_output_axes(selected_layers, selected_heads, aggregate)

    return AttentionOutput(
        attentions=attention_tensor,
        tokens=_normalize_tokens(tokens, attention_tensor.shape[-1]),
        layers=output_layers,
        heads=output_heads,
        aggregation=aggregate,
    )


def stack_attentions(attentions: tuple[Tensor, ...] | list[Tensor] | Tensor | None) -> Tensor:
    if attentions is None:
        raise ValueError("Attention maps are required")
    if isinstance(attentions, Tensor):
        return attentions
    if not attentions:
        raise ValueError("Model returned an empty attentions tuple")
    return torch.stack(tuple(attentions), dim=0)


def select_attentions(
    attentions: Tensor,
    *,
    layers: int | Sequence[int] | None = None,
    heads: int | Sequence[int] | None = None,
) -> tuple[Tensor, list[int], list[int]]:
    if attentions.ndim != 5:
        raise ValueError(
            "Raw attention tensors must have shape (layers, batch, heads, query_length, key_length), "
            f"got {tuple(attentions.shape)}"
        )

    selected_layers = _normalize_indices(layers, attentions.shape[0], "layers")
    selected_heads = _normalize_indices(heads, attentions.shape[2], "heads")
    attentions = attentions[selected_layers]
    attentions = attentions[:, :, selected_heads]
    return attentions, selected_layers, selected_heads


def aggregate_attentions(attentions: Tensor, aggregate: AttentionAggregation | None = None) -> Tensor:
    if aggregate is None:
        return attentions
    if attentions.ndim != 5:
        raise ValueError(
            "Attention aggregation expects raw stacked attention tensors with shape "
            "(layers, batch, heads, query_length, key_length)"
        )
    if aggregate == "mean_heads":
        return attentions.mean(dim=2)
    if aggregate == "mean_layers":
        return attentions.mean(dim=0)
    if aggregate == "rollout":
        return _attention_rollout(attentions)
    raise ValueError(f"Unsupported attention aggregation: {aggregate}")


def _resolve_output_axes(
    layers: list[int], heads: list[int], aggregate: AttentionAggregation | None
) -> tuple[list[int] | None, list[int] | None]:
    if aggregate is None:
        return layers, heads
    if aggregate == "mean_heads":
        return layers, None
    if aggregate in {"mean_layers", "rollout"}:
        return None, None if aggregate == "rollout" else heads
    raise ValueError(f"Unsupported attention aggregation: {aggregate}")


def _normalize_indices(indices: int | Sequence[int] | None, size: int, name: str) -> list[int]:
    if indices is None:
        return list(range(size))
    if isinstance(indices, int):
        normalized = [indices]
    else:
        normalized = list(indices)
    if not normalized:
        raise ValueError(f"{name} must contain at least one index")
    for index in normalized:
        if index < 0 or index >= size:
            raise ValueError(f"{name} index {index} is out of range for size {size}")
    return normalized


def _attention_rollout(attentions: Tensor) -> Tensor:
    rollout = attentions.mean(dim=2)
    identity = torch.eye(rollout.shape[-1], device=rollout.device, dtype=rollout.dtype)
    rollout = rollout + identity.view(1, 1, rollout.shape[-1], rollout.shape[-1])
    rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(rollout.dtype).eps)

    result = rollout[0]
    for layer_attention in rollout[1:]:
        result = torch.bmm(layer_attention, result)
    return result


def _normalize_tokens(tokens: Sequence[str] | None, length: int) -> list[str] | None:
    if tokens is None:
        return None
    if len(tokens) != length:
        raise ValueError(f"Expected {length} tokens, got {len(tokens)}")
    return [str(token) for token in tokens]


__all__ = [
    "AttentionAggregation",
    "aggregate_attentions",
    "prepare_attention_output",
    "select_attentions",
    "stack_attentions",
]
