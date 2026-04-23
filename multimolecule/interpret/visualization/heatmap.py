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

from typing import Sequence

import torch
from lazy_imports import try_import
from torch import Tensor

from ..outputs import AttentionOutput, AttributionOutput, SaeOutput

with try_import() as _matplotlib_import:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes


def plot_token_scores(
    output: AttributionOutput | Tensor,
    *,
    tokens: Sequence[str] | None = None,
    batch_index: int = 0,
    ax: Axes | None = None,
    cmap: str = "coolwarm",
    title: str | None = None,
) -> Axes:
    _matplotlib_import.check()
    token_scores = output.token_attributions if isinstance(output, AttributionOutput) else output
    token_scores = _to_tensor(token_scores)
    if token_scores.ndim == 2:
        token_scores = token_scores[batch_index]
    if token_scores.ndim != 1:
        raise ValueError(f"token scores must have shape (length,) or (batch, length), got {tuple(token_scores.shape)}")

    image = token_scores.unsqueeze(0)
    ax = _prepare_axis(ax)
    im = ax.imshow(image.numpy(), aspect="auto", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_yticks([])
    ax.set_xticks(range(token_scores.shape[0]))
    ax.set_xticklabels(_normalize_tokens(tokens, token_scores.shape[0]), rotation=90)
    ax.set_title(title or "Token Scores")
    return ax


def plot_attention_map(
    output: AttentionOutput | Tensor,
    *,
    tokens: Sequence[str] | None = None,
    batch_index: int = 0,
    layer: int | None = None,
    head: int | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    title: str | None = None,
) -> Axes:
    _matplotlib_import.check()
    attentions = output.attentions if isinstance(output, AttentionOutput) else output
    aggregation = output.aggregation if isinstance(output, AttentionOutput) else None
    if isinstance(output, AttentionOutput) and tokens is None:
        tokens = output.tokens
    attention_map = _select_attention_map(
        _to_tensor(attentions), aggregation, batch_index=batch_index, layer=layer, head=head
    )

    ax = _prepare_axis(ax)
    im = ax.imshow(attention_map.numpy(), aspect="auto", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = _normalize_tokens(tokens, attention_map.shape[-1])
    ax.set_xticks(range(attention_map.shape[-1]))
    ax.set_yticks(range(attention_map.shape[-2]))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title or "Attention Map")
    return ax


def plot_sae_features(
    output: SaeOutput | Tensor,
    *,
    batch_index: int = 0,
    feature_ids: Sequence[int | str] | Tensor | None = None,
    top_k: int | None = None,
    ax: Axes | None = None,
    cmap: str = "magma",
    title: str | None = None,
) -> Axes:
    _matplotlib_import.check()
    features = output.features if isinstance(output, SaeOutput) else output
    if isinstance(output, SaeOutput) and feature_ids is None:
        feature_ids = output.feature_ids
    features = _to_tensor(features)
    if features.ndim != 3:
        raise ValueError(f"SAE features must have shape (batch, length, features), got {tuple(features.shape)}")

    feature_map = features[batch_index].transpose(0, 1)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        top_k = min(top_k, feature_map.shape[0])
        importance = feature_map.abs().max(dim=1).values
        selected = torch.topk(importance, k=top_k).indices
        feature_map = feature_map.index_select(0, selected)
        if feature_ids is not None:
            feature_ids = _normalize_feature_ids(feature_ids, features.shape[-1])
            feature_ids = [feature_ids[index] for index in selected.tolist()]
    elif feature_ids is not None:
        feature_ids = _normalize_feature_ids(feature_ids, features.shape[-1])

    ax = _prepare_axis(ax)
    im = ax.imshow(feature_map.numpy(), aspect="auto", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(feature_map.shape[-1]))
    ax.set_yticks(range(feature_map.shape[0]))
    ax.set_yticklabels(feature_ids or [str(index) for index in range(feature_map.shape[0])])
    ax.set_title(title or "SAE Features")
    return ax


def _select_attention_map(
    attentions: Tensor,
    aggregation: str | None,
    *,
    batch_index: int,
    layer: int | None,
    head: int | None,
) -> Tensor:
    if attentions.ndim == 5:
        if layer is None or head is None:
            raise ValueError("Raw attention tensors require both layer and head indices for visualization")
        return attentions[layer, batch_index, head]
    if attentions.ndim == 4:
        if aggregation == "mean_heads":
            if layer is None:
                raise ValueError("mean_heads attention tensors require a layer index for visualization")
            return attentions[layer, batch_index]
        if aggregation == "mean_layers":
            if head is None:
                raise ValueError("mean_layers attention tensors require a head index for visualization")
            return attentions[batch_index, head]
        raise ValueError("4D attention tensors require aggregation metadata to choose the correct slice")
    if attentions.ndim == 3:
        return attentions[batch_index]
    raise ValueError(f"Unsupported attention tensor shape for visualization: {tuple(attentions.shape)}")


def _normalize_tokens(tokens: Sequence[str] | None, length: int) -> list[str]:
    if tokens is None:
        return [str(index) for index in range(length)]
    if len(tokens) != length:
        raise ValueError(f"Expected {length} tokens, got {len(tokens)}")
    return [str(token) for token in tokens]


def _normalize_feature_ids(feature_ids: Sequence[int | str] | Tensor, feature_size: int) -> list[str]:
    if isinstance(feature_ids, Tensor):
        feature_ids = feature_ids.detach().cpu().flatten().tolist()
    if len(feature_ids) != feature_size:
        raise ValueError(f"Expected {feature_size} feature ids, got {len(feature_ids)}")
    return [str(feature_id) for feature_id in feature_ids]


def _to_tensor(tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(tensor)!r}")
    return tensor.detach().cpu()


def _prepare_axis(ax: Axes | None) -> Axes:
    _matplotlib_import.check()
    if ax is not None:
        return ax
    _, ax = plt.subplots()
    return ax


__all__ = ["plot_attention_map", "plot_sae_features", "plot_token_scores"]
