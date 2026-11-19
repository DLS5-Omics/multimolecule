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

"""Inspect attention weights returned by PyTorch models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .._model import ModelInputs, OutputSelector, call_model, select_output

__all__ = ["capture"]


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
