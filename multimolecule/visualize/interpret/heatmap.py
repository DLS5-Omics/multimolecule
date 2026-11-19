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

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from torch import Tensor

from multimolecule.interpret.sae import SparseAutoencoderOutput


def plot_token_scores(
    scores: Tensor,
    *,
    tokens: Sequence[str] | None = None,
    batch_index: int = 0,
    ax: Axes | None = None,
    cmap: str = "coolwarm",
    title: str | None = None,
) -> Axes:
    """Plot one score per token from a tensor shaped ``(length,)`` or ``(batch, length)``.

    When passing batched scores, ``batch_index`` selects the sample. Reduce any
    embedding or channel dimensions before calling this function.
    """
    scores = _cpu_values(scores)
    if scores.ndim == 2:
        scores = scores[batch_index]
    if scores.ndim != 1:
        raise ValueError(f"Token scores must have shape (length,) or (batch, length), got {tuple(scores.shape)}.")

    labels = _labels(tokens, len(scores), "tokens")
    ax = _heatmap(scores.unsqueeze(0), ax=ax, cmap=cmap, title="Token Scores" if title is None else title)
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    return ax


def plot_attention_map(
    attention: Tensor,
    *,
    tokens: Sequence[str] | None = None,
    query_tokens: Sequence[str] | None = None,
    key_tokens: Sequence[str] | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    title: str | None = None,
) -> Axes:
    """Plot a selected 2D attention matrix shaped ``(queries, keys)``.

    Select the desired layer, sample and head before plotting. Pass ``tokens``
    for shared labels on both axes or pass separate query and key labels.
    """
    attention = _cpu_values(attention)
    if attention.ndim != 2:
        raise ValueError(f"Attention must be a selected 2D (queries, keys) matrix, got {tuple(attention.shape)}.")
    if tokens is not None:
        if query_tokens is not None or key_tokens is not None:
            raise ValueError("Pass tokens or separate query and key tokens.")
        query_tokens = key_tokens = tokens

    query_labels = _labels(query_tokens, attention.shape[0], "query tokens")
    key_labels = _labels(key_tokens, attention.shape[1], "key tokens")
    ax = _heatmap(attention, ax=ax, cmap=cmap, title="Attention Map" if title is None else title)
    ax.set_xticks(range(len(key_labels)), key_labels, rotation=90)
    ax.set_yticks(range(len(query_labels)), query_labels)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    return ax


def plot_jacobian(
    matrix: Tensor,
    *,
    input_labels: Sequence[int | str] | None = None,
    output_labels: Sequence[int | str] | None = None,
    ax: Axes | None = None,
    cmap: str = "coolwarm",
    title: str | None = None,
) -> Axes:
    """Plot a selected 2D Jacobian with output rows and input columns.

    Select or project batch and other dimensions before plotting. The color
    limits are symmetric around zero and span the largest finite derivative
    magnitude. Nonfinite entries are masked.
    """
    matrix = _cpu_values(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Jacobian must be a selected 2D (outputs, inputs) matrix, got {tuple(matrix.shape)}.")

    rows = _labels(output_labels, matrix.shape[0], "output labels")
    columns = _labels(input_labels, matrix.shape[1], "input labels")
    limit = matrix.masked_fill(~matrix.isfinite(), 0).abs().amax().item() or 1.0
    ax = _heatmap(
        matrix, ax=ax, cmap=cmap, title="Jacobian" if title is None else title, norm=Normalize(vmin=-limit, vmax=limit)
    )
    ax.set_xticks(range(len(columns)), columns, rotation=90)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    return ax


def plot_sae_features(
    features: SparseAutoencoderOutput | Tensor,
    *,
    batch_index: int = 0,
    feature_ids: Sequence[int | str] | Tensor | None = None,
    top_k: int | None = None,
    ax: Axes | None = None,
    cmap: str = "magma",
    title: str | None = None,
) -> Axes:
    """Plot SAE features across positions, with one feature per row.

    Features must have shape ``(positions, features)`` or
    ``(batch, positions, features)``. A ``SparseAutoencoderOutput`` supplies
    its ``features`` tensor. Select a batch entry with ``batch_index``.
    With ``top_k``, rank features by maximum absolute activation across positions.
    """
    values = _cpu_values(features.features if isinstance(features, SparseAutoencoderOutput) else features)
    if values.ndim == 3:
        values = values[batch_index]
    if values.ndim != 2:
        raise ValueError(
            f"SAE features must have shape (positions, features) or (batch, positions, features), "
            f"got {tuple(values.shape)}."
        )

    if isinstance(feature_ids, Tensor):
        feature_ids = feature_ids.detach().cpu().flatten().tolist()
    labels = _labels(feature_ids, values.shape[-1], "feature IDs")
    feature_map = values.transpose(0, 1)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        indices = feature_map.abs().amax(dim=1).topk(min(top_k, len(labels))).indices
        feature_map = feature_map.index_select(0, indices)
        labels = [labels[index] for index in indices.tolist()]

    ax = _heatmap(feature_map, ax=ax, cmap=cmap, title="SAE Features" if title is None else title)
    ax.set_xticks(range(feature_map.shape[-1]))
    ax.set_yticks(range(len(labels)), labels)
    return ax


def _cpu_values(values: Tensor) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(values).__name__}.")
    if values.numel() == 0:
        raise ValueError("Plotting requires a nonempty tensor.")
    values = values.detach().cpu()
    if values.is_floating_point() and values.element_size() < 4:
        values = values.float()
    return values


def _labels(labels: Sequence[int | str] | None, size: int, name: str) -> list[str]:
    if labels is None:
        return [str(index) for index in range(size)]
    if len(labels) != size:
        raise ValueError(f"Expected {size} {name}, got {len(labels)}.")
    return [str(label) for label in labels]


def _heatmap(values: Tensor, *, ax: Axes | None, cmap: str, title: str, norm: Normalize | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    image = ax.imshow(values.numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return ax
