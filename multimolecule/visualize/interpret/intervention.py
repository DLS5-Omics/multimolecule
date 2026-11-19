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

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor

from multimolecule.interpret.intervention import InterventionResult

from .heatmap import _cpu_values, _labels


def plot_intervention_deltas(
    deltas: Mapping[str, int | float | Tensor | InterventionResult],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot signed score changes for named intervention conditions.

    Supply one real number, one-element tensor, or ``InterventionResult`` with one
    selected delta per condition. Score changes are intervened minus original.
    Conditions appear in mapping order, with symmetric limits about zero.
    """
    if not deltas:
        raise ValueError("Provide at least one intervention condition.")

    values = []
    for name, supplied in deltas.items():
        delta = supplied.delta if isinstance(supplied, InterventionResult) else supplied
        if isinstance(delta, Tensor):
            if delta.numel() != 1:
                raise ValueError(f"Intervention {name!r} requires one selected score.")
            delta = _cpu_values(delta).item()
        values.append(_real_score(delta))

    if ax is None:
        _, ax = plt.subplots()
    positions = range(len(values))
    ax.barh(positions, values, color=["tab:blue" if value >= 0 else "tab:red" for value in values])
    ax.axvline(0, color="0.3", linewidth=1)
    limit = max(abs(value) for value in values) or 1.0
    ax.set_xlim(-limit, limit)
    ax.set_yticks(positions, list(deltas))
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("Intervened minus original score")
    ax.set_title("Intervention Deltas" if title is None else title)
    return ax


def plot_intervention_comparison(
    result: InterventionResult,
    *,
    sample_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot paired original and intervened scores on a shared axis.

    Select one score per sample in the intervention comparison. The original
    and intervened tensors must be scalars or matching one-dimensional batches.
    """
    original, intervened = result.original, result.intervened
    if original.shape != intervened.shape or original.ndim > 1:
        raise ValueError("Original and intervened outputs must be matching scalar or 1D score tensors.")
    before = _score_list(original)
    after = _score_list(intervened)
    labels = _labels(sample_labels, len(before), "sample labels")

    if ax is None:
        _, ax = plt.subplots()
    positions = range(len(labels))
    width = 0.4
    ax.bar([position - width / 2 for position in positions], before, width, color="tab:blue", label="Original")
    ax.bar([position + width / 2 for position in positions], after, width, color="tab:orange", label="Intervened")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Selected score")
    ax.set_title("Intervention Comparison" if title is None else title)
    ax.legend()
    return ax


def _score_list(scores: Tensor) -> list[float]:
    return [_real_score(value) for value in _cpu_values(scores).reshape(-1).tolist()]


def _real_score(value: object) -> float:
    if not isinstance(value, Real):
        raise TypeError("Intervention scores must be real numbers.")
    score = float(value)
    if not isfinite(score):
        raise ValueError("Intervention scores must be finite.")
    return score
