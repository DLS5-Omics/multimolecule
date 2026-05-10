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

from typing import Any, Mapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from multimolecule.utils.rna.secondary_structure import pseudoknot_tiers
from multimolecule.utils.rna.secondary_structure.notations import dot_bracket_to_pairs

from ..palettes import LOOP_TYPE_PALETTE, PAIR_COMPARISON_COLORS, PSEUDOKNOT_TIER_COLORS
from .comparison import compare_secondary_structures
from .layout import _primary_layout_pairs
from .tracks import BaseCategoryTrack, BaseValueTrack, ColorTrack, RegionTrack

Category = Union[str, int]
Pair = tuple[int, int]

_DEFAULT_REGION_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_COLORBAR_GID = "multimolecule-attached-colorbar"
_COLORBAR_WIDTH = 0.025
_COLORBAR_SLOT_STEP = 0.075
_PANEL_TITLE_X = 0.5
_PANEL_TITLE_Y = 0.99
_PANEL_TITLE_FONTSIZE = 12
_PANEL_TITLE_COLOR = "#111111"
_PANEL_TITLE_ZORDER = 10
_LW_EDGE_TO_MARKER: dict[str, str] = {"WC": "o", "H": "s", "S": "^"}


def _create_panel_axes(*, figsize: tuple[float, float]):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    return fig, ax


def _fill_panel_axes(ax: Axes) -> None:
    ax.set_position((0.0, 0.0, 1.0, 1.0))


def _add_panel_title(ax: Axes, title: str | None) -> None:
    if title:
        ax.text(
            _PANEL_TITLE_X,
            _PANEL_TITLE_Y,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=_PANEL_TITLE_FONTSIZE,
            color=_PANEL_TITLE_COLOR,
            zorder=_PANEL_TITLE_ZORDER,
        )


def _default_bead_size(length: int) -> float:
    if length <= 80:
        return 24.0
    if length <= 180:
        return 18.0
    if length <= 240:
        return 12.0
    if length <= 500:
        return 8.0
    return 6.0


def _default_structure_figsize(length: int) -> float:
    if length <= 180:
        return 5.0
    if length <= 400:
        return 6.5
    return 7.5


def _add_attached_colorbar(fig: Figure, ax: Axes, mappable, *, label: str | None = None):
    """Add a colorbar inside the plotted axes to avoid expanding the figure footprint."""
    slot = sum(child.get_gid() == _COLORBAR_GID for child in ax.child_axes)
    left = 1.0 - _COLORBAR_WIDTH - slot * _COLORBAR_SLOT_STEP
    cax = ax.inset_axes([left, 0.0, _COLORBAR_WIDTH, 1.0], transform=ax.transAxes)
    cax.set_gid(_COLORBAR_GID)
    return fig.colorbar(mappable, cax=cax, label=label)


def _check_sequence_length(sequence: str | None, length: int, *, target: str = "structure") -> None:
    if sequence is not None and len(sequence) != length:
        raise ValueError(f"sequence length {len(sequence)} != {target} length {length}.")


def _check_reference_dot_bracket_length(reference_dot_bracket: str | None, length: int) -> None:
    if reference_dot_bracket is not None and len(reference_dot_bracket) != length:
        raise ValueError(f"reference_dot_bracket length {len(reference_dot_bracket)} != dot_bracket length {length}.")


def _resolve_show_bases(sequence: str | None, show_bases: bool | None, length: int, *, max_length: int) -> bool:
    if show_bases is None:
        return sequence is not None and length <= max_length
    if show_bases and sequence is None:
        raise ValueError("sequence is required when show_bases=True.")
    return show_bases


def _value_track_rgba(track: BaseValueTrack) -> tuple[np.ndarray, Any]:
    """Resolve a [BaseValueTrack][] to an Nx4 RGBA array and a ScalarMappable for colorbars."""
    values = track.values_array()
    finite = np.isfinite(values)
    if finite.any():
        data_min = float(np.nanmin(values))
        data_max = float(np.nanmax(values))
    else:
        data_min, data_max = 0.0, 1.0
    vmin = track.vmin if track.vmin is not None else data_min
    vmax = track.vmax if track.vmax is not None else data_max
    if vmax <= vmin:
        # Avoid a degenerate Normalize range (matplotlib divides by vmax - vmin).
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps[track.cmap]
    rgba = cmap(norm(np.where(finite, values, vmin)))
    rgba[~finite] = (0.65, 0.65, 0.65, 1.0)
    return rgba, ScalarMappable(norm=norm, cmap=cmap)


def _category_palette(track: BaseCategoryTrack) -> dict[Category, str]:
    """Return a category -> color mapping, falling back to LOOP_TYPE_PALETTE + cycle defaults."""
    if track.palette is not None:
        return dict(track.palette)
    seen: list[Category] = []
    for category in track.categories:
        if category not in seen:
            seen.append(category)
    palette: dict[Category, str] = {}
    fallback_index = 0
    for category in seen:
        key = category.lower() if isinstance(category, str) else category
        if isinstance(key, str) and key in LOOP_TYPE_PALETTE:
            palette[category] = LOOP_TYPE_PALETTE[key]
        else:
            palette[category] = _DEFAULT_REGION_PALETTE[fallback_index % len(_DEFAULT_REGION_PALETTE)]
            fallback_index += 1
    return palette


def _category_track_colors(track: BaseCategoryTrack) -> tuple[list[str], dict[Category, str]]:
    palette = _category_palette(track)
    colors = [palette.get(category, "#999999") for category in track.categories]
    return colors, palette


def _color_track_rgba(track: ColorTrack) -> np.ndarray:
    """Resolve any color-bearing track to an Nx4 RGBA float array."""
    if isinstance(track, BaseValueTrack):
        rgba, _ = _value_track_rgba(track)
        return rgba
    colors, _ = _category_track_colors(track)
    return np.asarray([to_rgba(color) for color in colors], dtype=float)


def _resolve_bead_colors(
    sequence: str | None,
    length: int,
    palette_colors: Mapping[str, str],
    color_track: ColorTrack | None,
) -> list[Any]:
    if color_track is not None:
        rgba = _color_track_rgba(color_track)
        return [tuple(row) for row in rgba.tolist()]
    return [
        (
            palette_colors.get(sequence[i].upper(), palette_colors["N"])
            if sequence and i < len(sequence)
            else palette_colors["N"]
        )
        for i in range(length)
    ]


def _maybe_add_color_track_legend(
    fig: Figure,
    ax: Axes,
    color_track: ColorTrack | None,
) -> None:
    if color_track is None:
        return
    if isinstance(color_track, BaseValueTrack):
        if not color_track.show_colorbar:
            return
        _, mappable = _value_track_rgba(color_track)
        mappable.set_array(np.asarray(color_track.values, dtype=float))
        _add_attached_colorbar(fig, ax, mappable, label=color_track.name)
        return
    if not color_track.show_legend:
        return

    _, palette = _category_track_colors(color_track)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="#333333",
            linestyle="",
            label=str(label),
        )
        for label, color in palette.items()
    ]
    if handles:
        previous_legend = ax.get_legend()
        ax.legend(
            handles=handles,
            title=color_track.name,
            loc="upper right",
            frameon=False,
            fontsize=7,
            title_fontsize=8,
            handlelength=1.0,
        )
        if previous_legend is not None:
            ax.add_artist(previous_legend)


def _region_colors(track: RegionTrack, offset: int = 0) -> list[str]:
    if isinstance(track.colors, str):
        return [track.colors] * len(track.regions)
    if track.colors is None:
        return [_DEFAULT_REGION_PALETTE[(offset + i) % len(_DEFAULT_REGION_PALETTE)] for i in range(len(track.regions))]
    colors_list = list(track.colors)
    if not colors_list:
        return [_DEFAULT_REGION_PALETTE[(offset + i) % len(_DEFAULT_REGION_PALETTE)] for i in range(len(track.regions))]
    return [colors_list[i % len(colors_list)] for i in range(len(track.regions))]


def _pair_tier_color_map(
    pairs: np.ndarray,
    layout_pair_set: set[Pair] | None,
    default_color: str,
    show_pseudoknot_tiers: bool,
) -> dict[Pair, str]:
    colors: dict[Pair, str] = {}
    pair_list = [(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs.tolist()]
    if not pair_list:
        return colors

    if not show_pseudoknot_tiers:
        for key in pair_list:
            colors[key] = default_color
        return colors

    if layout_pair_set is not None:
        pair_set = set(pair_list)
        primary_pairs = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in layout_pair_set}
        non_primary_pairs = [pair for pair in pair_list if pair not in primary_pairs]
        for key in primary_pairs:
            # Public callers can supply an external layout_pair_set, so keep the membership guard.
            if key in pair_set:
                colors[key] = default_color
        if non_primary_pairs:
            non_primary_array = np.asarray(non_primary_pairs, dtype=int)
            tiers = pseudoknot_tiers(non_primary_array)
            for tier_index, tier_pairs in enumerate(tiers):
                color = PSEUDOKNOT_TIER_COLORS[tier_index % len(PSEUDOKNOT_TIER_COLORS)]
                for i, j in tier_pairs.tolist():
                    colors[(int(i), int(j))] = color
    else:
        tiers = pseudoknot_tiers(np.asarray(pair_list, dtype=int))
        for tier_index, tier_pairs in enumerate(tiers):
            if tier_index == 0:
                color = default_color
            else:
                color = PSEUDOKNOT_TIER_COLORS[(tier_index - 1) % len(PSEUDOKNOT_TIER_COLORS)]
            for i, j in tier_pairs.tolist():
                colors[(int(i), int(j))] = color

    for key in pair_list:
        colors.setdefault(key, default_color)
    return colors


def _primary_pair_set(dot_bracket: str, pairs: np.ndarray) -> set[Pair]:
    return {(min(int(i), int(j)), max(int(i), int(j))) for i, j in _primary_layout_pairs(dot_bracket, pairs).tolist()}


def _structure_preflight(
    dot_bracket: str,
    sequence: str | None,
    reference_dot_bracket: str | None,
) -> tuple[int, np.ndarray]:
    """Validate dot-bracket inputs and return ``(length, pairs)``. Common entry guard for plot_* functions."""
    if len(dot_bracket) == 0:
        raise ValueError("dot_bracket is empty.")
    length = len(dot_bracket)
    _check_sequence_length(sequence, length)
    _check_reference_dot_bracket_length(reference_dot_bracket, length)
    return length, dot_bracket_to_pairs(dot_bracket)


def _resolve_panel_axes(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Figure, Axes, bool]:
    """Resolve the (fig, ax, created_ax) triple shared by every plot_* entry point."""
    if ax is None:
        fig, ax = _create_panel_axes(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _comparison_pair_specs(
    dot_bracket: str,
    reference_dot_bracket: str,
    *,
    fn_linewidth: float,
    tp_linewidth: float,
    fn_alpha: float,
    tp_alpha: float,
) -> list[tuple[int, int, str, float, float, str]]:
    """Build pair specs colored by TP/FP/FN classification for an overlay comparison rendering.

    FN pairs are dashed; TP and FP are solid. Per-renderer width/alpha are exposed so the
    three orchestrators can keep their visual tuning while sharing the loop body.
    """
    comparison = compare_secondary_structures(dot_bracket, reference_dot_bracket)
    specs: list[tuple[int, int, str, float, float, str]] = [
        (i, j, PAIR_COMPARISON_COLORS["false_negative"], fn_linewidth, fn_alpha, "--")
        for i, j in comparison["false_negative_pairs"]
    ]
    specs += [
        (i, j, PAIR_COMPARISON_COLORS["true_positive"], tp_linewidth, tp_alpha, "-")
        for i, j in comparison["true_positive_pairs"]
    ]
    specs += [
        (i, j, PAIR_COMPARISON_COLORS["false_positive"], tp_linewidth, tp_alpha, "-")
        for i, j in comparison["false_positive_pairs"]
    ]
    return specs


def _draw_lw_glyphs(
    ax: Axes,
    *,
    points: Sequence[tuple[float, float]],
    edge: str,
    orientation: str,
    size: float,
    color: str,
) -> None:
    """Render Leontis-Westhof edge glyphs. cis pairs use a filled marker; trans pairs use an open marker."""
    if not points:
        return
    marker = _LW_EDGE_TO_MARKER[edge]
    face = color if orientation == "cis" else "white"
    xy = np.asarray(points, dtype=float)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=size,
        marker=marker,
        facecolors=face,
        edgecolors=color,
        linewidths=0.9,
        zorder=4,
    )
