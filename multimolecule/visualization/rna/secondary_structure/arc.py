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

from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from multimolecule.utils.rna.secondary_structure.notations import dot_bracket_to_pairs

from ..palettes import resolve_nucleotide_colors
from .comparison import _add_comparison_legend
from .probability import (
    _DEFAULT_MFE_PROBABILITY_ALPHA,
    _DEFAULT_MFE_PROBABILITY_COLOR,
    _DEFAULT_MFE_PROBABILITY_LINEWIDTH,
    _PROBABILITY_HAZE_LINEWIDTH,
    _collect_probability_tracks,
    _maybe_add_probability_colorbar,
    _probability_alpha,
    _probability_color_and_width,
    _probability_haze_pairs,
    _probability_pairs,
)
from .tracks import (
    ColorTrack,
    PairProbabilityTrack,
    RegionTrack,
    Track,
    _collect_color_track,
    _collect_region_tracks,
    _collect_sequence_diff_track,
    _sequence_diff_mask,
)
from .utils import (
    _add_panel_title,
    _color_track_rgba,
    _comparison_pair_specs,
    _fill_panel_axes,
    _maybe_add_color_track_legend,
    _pair_tier_color_map,
    _primary_pair_set,
    _region_colors,
    _resolve_panel_axes,
    _resolve_show_bases,
    _structure_preflight,
)

_ARC_MIN_FIGURE_HEIGHT = 1.0


def plot_arc_diagram(
    dot_bracket: str,
    sequence: str | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_bases: bool | None = None,
    tracks: Sequence[Track] | None = None,
    reference_dot_bracket: str | None = None,
    comparison_style: str = "overlay",
    show_pseudoknot_tiers: bool = True,
    pair_color: str = "#2563eb",
    reference_pair_color: str = "#666666",
    backbone_color: str = "#404040",
    arc_height_scale: float = 1.0,
    base_font_size: float = 6.0,
    palette: str = "nature",
    base_colors: Mapping[str, str] | None = None,
) -> Figure:
    """
    Plot a dot-bracket RNA secondary structure as an arc diagram.

    Args:
        dot_bracket: Dot-bracket notation.
        sequence: RNA sequence. It is only used for optional base labels. Use ``None`` when sequence information is
            not available.
        ax: Optional matplotlib axes.
        title: Plot title.
        show_bases: Whether to render base labels. Defaults to true for structures up to 160 nt.
        tracks: Optional sequence of overlay tracks. [BaseValueTrack][] and
            [BaseCategoryTrack][] render as colored markers along the baseline;
            [RegionTrack][] renders as an x-range band with a label;
            [PairProbabilityTrack][] renders probability arcs; and
            [SequenceDiffTrack][] emphasizes changed base labels.
            [PairAnnotationTrack][] is ignored on arc diagrams.
        reference_dot_bracket: Optional reference dot-bracket. When provided, see
            ``comparison_style`` for how the comparison is drawn.
        comparison_style: ``"overlay"`` colors arcs as TP / FP / FN on a single set of arcs
            above the baseline. ``"dual"`` draws the predicted structure above and the reference
            below the baseline (R-chie idiom).
        show_pseudoknot_tiers: When ``True``, pairs from non-primary brackets are colored from
            [PSEUDOKNOT_TIER_COLORS][] so crossing tiers are visually distinguishable. When
            ``False`` every pair uses ``pair_color`` regardless of nesting tier.
        pair_color: Predicted-arc color (used when there is no reference, or in ``"dual"`` mode).
        reference_pair_color: Reference-arc color used in ``"dual"`` mode below the baseline.
        backbone_color: Backbone line color.
        arc_height_scale: Vertical scale for arc height. ``1.0`` gives classical semicircles.
            Values above 1.0 produce taller arcs; smaller values flatten arcs. The axes keep
            equal data-unit geometry, so this controls the rendered curvature directly.
        base_font_size: Font size for base labels.
        palette: Name from [NUCLEOTIDE_PALETTES][]. Base labels along the baseline are
            colored by nucleotide, matching [plot_planar_graph][] and [plot_circular_diagram][].
        base_colors: Optional per-base color overrides for label coloring. Forwarded to
            [resolve_nucleotide_colors][].

    Returns:
        The matplotlib figure containing the plot.

    Raises:
        ValueError: If ``dot_bracket`` is empty, ``sequence`` or ``reference_dot_bracket``
            length disagrees with ``dot_bracket``, ``comparison_style`` is not
            ``"overlay"`` or ``"dual"``, or ``arc_height_scale`` is not a positive
            number ``<= 2.0``.

    Examples:
        >>> from multimolecule.visualization.rna import plot_arc_diagram
        >>> fig = plot_arc_diagram("(((...)))", "GGGAAAUCC")
        >>> type(fig).__name__
        'Figure'
    """
    if comparison_style not in {"overlay", "dual"}:
        raise ValueError(f"comparison_style must be 'overlay' or 'dual'; got {comparison_style!r}.")

    length, pairs = _structure_preflight(dot_bracket, sequence, reference_dot_bracket)
    is_dual = reference_dot_bracket is not None and comparison_style == "dual"
    probability_tracks = _collect_probability_tracks(tracks, length)
    height_scale = _arc_height_scale(arc_height_scale)
    fig_height = 5.2 if is_dual else 3.2
    fig_width = min(14.0, max(7.0, length / 32))
    fig, ax, created_ax = _resolve_panel_axes(ax, figsize=(fig_width, fig_height))

    region_tracks = _collect_region_tracks(tracks, length)
    _arc_render_regions(ax, region_tracks, length)

    ax.plot([0, length - 1], [0, 0], color=backbone_color, linewidth=1.0, zorder=2)

    color_source = _collect_color_track(tracks, length)
    _arc_render_color_markers(ax, color_source)

    diff_track = _collect_sequence_diff_track(tracks)
    diff_mask = _sequence_diff_mask(diff_track, sequence, length)

    show_bases = _resolve_show_bases(sequence, show_bases, length, max_length=160)
    if show_bases:
        assert sequence is not None
        label_colors = resolve_nucleotide_colors(palette, base_colors)
        for index, base in enumerate(sequence[:length]):
            base_upper = base.upper()
            if diff_track is not None and diff_mask[index]:
                color = diff_track.highlight_color
            else:
                color = label_colors.get(base_upper, label_colors["N"])
            weight = "bold" if (diff_track is not None and diff_track.bold_label and diff_mask[index]) else "normal"
            ax.text(
                index,
                -0.12,
                base_upper,
                ha="center",
                va="top",
                fontsize=base_font_size,
                color=color,
                fontweight=weight,
            )
    elif length > 1:
        ax.text(0, -0.12, "5'", ha="left", va="top", fontsize=8)
        ax.text(length - 1, -0.12, "3'", ha="right", va="top", fontsize=8)

    max_top = 0.0
    max_bottom = 0.0

    max_top = max(max_top, _arc_render_probability(ax, probability_tracks, sign=1, height_scale=height_scale))

    pair_tier_colors = _pair_tier_color_map(
        pairs,
        _primary_pair_set(dot_bracket, pairs),
        pair_color,
        show_pseudoknot_tiers,
    )
    if reference_dot_bracket is None and not probability_tracks:
        specs = [(int(i), int(j), pair_tier_colors[(int(i), int(j))], 1.15, 0.82, "-") for i, j in pairs.tolist()]
        max_top = max(max_top, _draw_arc_collection(ax, specs, sign=1, height_scale=height_scale))
    elif reference_dot_bracket is None:
        # Probability arcs are the main rendering; draw MFE arcs as a quiet reference layer.
        specs = [
            (
                int(i),
                int(j),
                _DEFAULT_MFE_PROBABILITY_COLOR,
                _DEFAULT_MFE_PROBABILITY_LINEWIDTH,
                _DEFAULT_MFE_PROBABILITY_ALPHA,
                "-",
            )
            for i, j in pairs.tolist()
        ]
        max_top = max(max_top, _draw_arc_collection(ax, specs, sign=1, height_scale=height_scale))
    elif comparison_style == "overlay":
        specs = _comparison_pair_specs(
            dot_bracket,
            reference_dot_bracket,
            fn_linewidth=1.0,
            tp_linewidth=1.2,
            fn_alpha=0.72,
            tp_alpha=0.9,
        )
        max_top = max(max_top, _draw_arc_collection(ax, specs, sign=1, height_scale=height_scale))
    else:
        specs = [(int(i), int(j), pair_color, 1.15, 0.82, "-") for i, j in pairs.tolist()]
        max_top = max(max_top, _draw_arc_collection(ax, specs, sign=1, height_scale=height_scale))
        ref_pairs = dot_bracket_to_pairs(reference_dot_bracket)
        specs = [(int(i), int(j), reference_pair_color, 1.15, 0.82, "-") for i, j in ref_pairs.tolist()]
        max_bottom = max(max_bottom, _draw_arc_collection(ax, specs, sign=-1, height_scale=height_scale))

    label_drop = -0.32 if region_tracks else -0.35
    if length == 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(0, length - 1)
    plot_max_top = max_top
    plot_max_bottom = max_bottom
    if is_dual:
        extent = max(plot_max_top, plot_max_bottom, 0.1)
        ax.set_ylim(-extent, extent)
        y_span = 2 * extent
    else:
        y_max = max(plot_max_top, 0.1)
        ax.set_ylim(label_drop, y_max)
        y_span = y_max - label_drop
    preserve_equal_aspect = True
    if created_ax:
        raw_fig_height = fig_width * y_span / max(length - 1, 1)
        fig_height = max(_ARC_MIN_FIGURE_HEIGHT, raw_fig_height)
        preserve_equal_aspect = raw_fig_height >= _ARC_MIN_FIGURE_HEIGHT
        fig.set_size_inches(fig_width, fig_height, forward=True)
        _fill_panel_axes(ax)
    ax.set_aspect("equal" if preserve_equal_aspect else "auto", adjustable="box")
    ax.axis("off")
    if reference_dot_bracket is not None and comparison_style == "overlay":
        _add_comparison_legend(ax)
    _maybe_add_color_track_legend(fig, ax, color_source)
    _maybe_add_probability_colorbar(fig, ax, probability_tracks)
    _add_panel_title(ax, title)
    return fig


def _arc_render_regions(ax: Axes, region_tracks: list[RegionTrack], length: int) -> None:
    offset = 0
    for track in region_tracks:
        colors = _region_colors(track, offset=offset)
        for (start, stop, label), color in zip(track.regions, colors):
            start_x = max(-0.5, float(start) - 0.5)
            stop_x = min(length - 0.5, float(stop) + 0.5)
            ax.axvspan(start_x, stop_x, ymin=0.0, ymax=0.04, color=color, alpha=track.alpha, zorder=1)
            if track.show_labels and label:
                ax.text(
                    (start_x + stop_x) / 2,
                    -0.22,
                    label,
                    color=color,
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    zorder=5,
                )
        offset += len(track.regions)


def _arc_render_color_markers(
    ax: Axes,
    color_track: ColorTrack | None,
) -> None:
    if color_track is None:
        return
    rgba = _color_track_rgba(color_track)
    positions = np.arange(len(rgba))
    ax.scatter(positions, np.zeros_like(positions, dtype=float), s=16, c=rgba, edgecolors="none", zorder=3)


ArcSpec = tuple[int, int, Any, float, float, str]


def _arc_segment(i: int, j: int, *, sign: int, height_scale: float) -> tuple[np.ndarray, float]:
    i, j = int(i), int(j)
    if i > j:
        i, j = j, i
    center = (i + j) / 2
    radius = (j - i) / 2
    height = radius * height_scale
    theta = np.linspace(0, np.pi, 100)
    xs = center + radius * np.cos(theta)
    ys = sign * height * np.sin(theta)
    return np.column_stack((xs, ys)), float(height)


def _draw_arc_collection(
    ax: Axes,
    specs: Sequence[ArcSpec],
    *,
    sign: int,
    height_scale: float,
    zorder: float = 3,
) -> float:
    if not specs:
        return 0.0

    max_height = 0.0
    by_style: dict[str, tuple[list[np.ndarray], list[Any], list[float]]] = {}
    for i, j, color, linewidth, alpha, linestyle in specs:
        segment, height = _arc_segment(i, j, sign=sign, height_scale=height_scale)
        max_height = max(max_height, height)
        segments, colors, linewidths = by_style.setdefault(linestyle, ([], [], []))
        segments.append(segment)
        colors.append(to_rgba(color, alpha=alpha))
        linewidths.append(linewidth)

    for linestyle, (segments, colors, linewidths) in by_style.items():
        ax.add_collection(
            LineCollection(
                segments,
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyle,
                zorder=zorder,
            )
        )
    return max_height


def _arc_height_scale(arc_height_scale: float) -> float:
    value = float(arc_height_scale)
    if not np.isfinite(value) or value <= 0.0 or value > 2.0:
        raise ValueError("arc_height_scale must be a positive number <= 2.0.")
    return value


def _arc_render_probability(
    ax: Axes,
    probability_tracks: list[PairProbabilityTrack],
    *,
    sign: int = 1,
    height_scale: float = 1.0,
) -> float:
    specs: list[ArcSpec] = []
    for track in probability_tracks:
        for i, j, p in _probability_haze_pairs(track):
            color, _ = _probability_color_and_width(track, p)
            specs.append((i, j, color, _PROBABILITY_HAZE_LINEWIDTH, 0.14, "-"))
        for i, j, p in _probability_pairs(track):
            color, width = _probability_color_and_width(track, p)
            specs.append((i, j, color, width, _probability_alpha(p), "-"))
    return _draw_arc_collection(ax, specs, sign=sign, height_scale=height_scale)
