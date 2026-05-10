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

from typing import Any, Sequence, TypeAlias

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .tracks import PairProbabilityTrack, Track
from .utils import _add_attached_colorbar

ProbabilityPair: TypeAlias = tuple[int, int, float]

_DEFAULT_MFE_PROBABILITY_COLOR = "#253247"
_DEFAULT_MFE_PROBABILITY_ALPHA = 0.34
_DEFAULT_MFE_PROBABILITY_LINEWIDTH = 0.45
_PROBABILITY_HAZE_LINEWIDTH = 0.25
_PROBABILITY_LINEWIDTH_MIN = 0.35
_PROBABILITY_LINEWIDTH_MAX = 1.05
_PROBABILITY_ALPHA_MIN = 0.28
_PROBABILITY_ALPHA_MAX = 0.72


def _collect_probability_tracks(
    tracks: Sequence[Track] | None,
    length: int,
) -> list[PairProbabilityTrack]:
    if not tracks:
        return []
    result: list[PairProbabilityTrack] = []
    for track in tracks:
        if isinstance(track, PairProbabilityTrack):
            track.validate(length)
            result.append(track)
    return result


def _probability_pairs(track: PairProbabilityTrack) -> list[ProbabilityPair]:
    return _probability_pairs_in_range(track, low=track.threshold, high=None)


def _probability_haze_pairs(track: PairProbabilityTrack) -> list[ProbabilityPair]:
    """Pairs in ``[haze_threshold, threshold)`` — the faint background layer."""
    if track.haze_threshold is None:
        return []
    return _probability_pairs_in_range(track, low=track.haze_threshold, high=track.threshold)


def _probability_pairs_in_range(
    track: PairProbabilityTrack,
    *,
    low: float,
    high: float | None,
) -> list[ProbabilityPair]:
    matrix = track.matrix_array()
    length = matrix.shape[0]
    if length < 2:
        return []
    rows, cols = np.triu_indices(length, k=1)
    values = matrix[rows, cols]
    mask = np.isfinite(values) & (values >= low)
    if high is not None:
        mask &= values < high
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    return list(zip(rows.tolist(), cols.tolist(), values.tolist()))


def _probability_color_and_width(track: PairProbabilityTrack, p: float) -> tuple[Any, float]:
    cmap = mpl.colormaps[track.cmap]
    probability = float(min(1.0, max(0.0, p)))
    color = cmap(probability)
    if track.width_by_probability:
        width = _PROBABILITY_LINEWIDTH_MIN + (_PROBABILITY_LINEWIDTH_MAX - _PROBABILITY_LINEWIDTH_MIN) * probability
    else:
        width = 0.65
    return color, width


def _probability_alpha(p: float) -> float:
    probability = float(min(1.0, max(0.0, p)))
    return _PROBABILITY_ALPHA_MIN + (_PROBABILITY_ALPHA_MAX - _PROBABILITY_ALPHA_MIN) * probability


def _maybe_add_probability_colorbar(
    fig: Figure,
    ax: Axes,
    probability_tracks: list[PairProbabilityTrack],
) -> None:
    if not probability_tracks:
        return
    track = probability_tracks[-1]

    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[track.cmap])
    mappable.set_array(np.asarray([0.0, 1.0], dtype=float))
    _add_attached_colorbar(fig, ax, mappable, label=track.name)
