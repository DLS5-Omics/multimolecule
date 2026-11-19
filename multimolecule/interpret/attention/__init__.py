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

"""Inspect and aggregate attention weights returned by PyTorch models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..utils import ModelInputs, OutputSelector, call_model, select_output

__all__ = ["capture", "mean_heads", "mean_layers", "rollout"]


def _select_heads(weights: Tensor, heads: Sequence[int], head_dim: int) -> Tensor:
    head_count = weights.size(head_dim)
    if any(not -head_count <= head < head_count for head in heads):
        raise IndexError(f"Head index out of range for an attention tensor with {head_count} heads.")
    indices = [head % head_count for head in heads]
    return weights.index_select(head_dim, torch.tensor(indices, dtype=torch.long, device=weights.device))


def capture(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    layers: int | Sequence[int] | None = None,
    heads: int | Sequence[int] | None = None,
    head_dim: int = 1,
    output: OutputSelector = "attentions",
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Tensor, ...]:
    """Collect native attention weights and optionally select layers and heads.

    The output selector should yield one tensor or a sequence of tensors in
    layer order. Pass ``model_kwargs={"output_attentions": True}`` for models
    that expose weights through that flag. Layer and head indices can be
    negative. Head selection preserves the head axis and all other axes.
    Select ``heads`` from weights with an explicit head axis. For example, request
    ``average_attn_weights=False`` from PyTorch's ``MultiheadAttention``.
    """
    value = select_output(call_model(model, inputs, model_kwargs), output)
    if value is None:
        raise ValueError(
            "The model did not return attention weights. Select an attention output or request "
            "output_attentions through model_kwargs for models supporting that flag."
        )
    if isinstance(value, Tensor):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("The selected attention output must be a tensor or a sequence of tensors.")
    if len(value) == 0:
        raise ValueError("The selected attention output is empty.")

    if isinstance(layers, int):
        layers = (layers,)
    weights = tuple(value) if layers is None else tuple(value[index] for index in layers)
    if any(not isinstance(layer_weights, Tensor) for layer_weights in weights):
        raise TypeError("Each selected layer must return an attention tensor.")
    if heads is None:
        return weights
    if isinstance(heads, int):
        heads = (heads,)
    return tuple(_select_heads(layer_weights, heads, head_dim) for layer_weights in weights)


def mean_heads(weights: Sequence[Tensor], *, head_dim: int = 1) -> tuple[Tensor, ...]:
    """Average the heads of each layer and preserve every other axis."""
    _validate_weights(weights)
    return tuple(_mean_head_axis(layer, head_dim) for layer in weights)


def mean_layers(weights: Sequence[Tensor]) -> Tensor:
    """Average same-shaped attention maps element by element across layers."""
    _validate_weights(weights)
    if any(layer.shape != weights[0].shape for layer in weights[1:]):
        raise ValueError("All attention layers must have the same shape for layer averaging.")
    return torch.stack(tuple(weights)).mean(dim=0)


def rollout(weights: Sequence[Tensor], *, head_dim: int = 1) -> Tensor:
    """Compose dense square attention maps in forward layer order.

    Each layer has shape ``(batch, heads, length, length)`` after moving its head axis
    to position 1. Rollout averages each layer's heads, adds the residual identity,
    and normalizes each row. Later layers multiply earlier layers from the left.
    The result has shape ``(batch, length, length)``.
    """
    _validate_weights(weights)
    if any(layer.ndim != 4 for layer in weights):
        raise ValueError("Rollout requires dense square attention maps with batch and head axes.")
    first, *remaining = (layer.movedim(head_dim, 1) for layer in weights)
    batch, _, query_length, key_length = first.shape
    if query_length == 0 or query_length != key_length:
        raise ValueError("Rollout requires nonempty square attention maps.")
    if any(layer.size(0) != batch or layer.shape[-2:] != first.shape[-2:] for layer in remaining):
        raise ValueError("All rollout layers must have the same batch and square attention dimensions.")
    if any(layer.device != first.device or layer.dtype != first.dtype for layer in remaining):
        raise ValueError("All rollout layers must have the same dtype and device.")

    identity = torch.eye(query_length, device=first.device, dtype=first.dtype)
    result = _residual_attention(first, identity)
    for layer in remaining:
        result = torch.bmm(_residual_attention(layer, identity), result)
    return result


def _validate_weights(weights: Sequence[Tensor]) -> None:
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
        raise TypeError("Pass a sequence of attention tensors in forward layer order.")
    if len(weights) == 0:
        raise ValueError("Pass at least one layer of attention weights.")
    if any(not isinstance(layer, Tensor) or not layer.is_floating_point() for layer in weights):
        raise TypeError("Attention weights must be floating-point tensors.")


def _mean_head_axis(layer: Tensor, head_dim: int) -> Tensor:
    if layer.size(head_dim) == 0:
        raise ValueError("Each attention layer must have at least one head.")
    return layer.mean(dim=head_dim)


def _residual_attention(layer: Tensor, identity: Tensor) -> Tensor:
    residual = _mean_head_axis(layer, 1) + identity
    return residual / residual.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(residual.dtype).eps)
