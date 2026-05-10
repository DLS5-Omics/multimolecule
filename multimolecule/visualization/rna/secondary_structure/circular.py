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

from typing import Callable, Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

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
    PairAnnotationTrack,
    PairProbabilityTrack,
    RegionTrack,
    Track,
    _collect_color_track,
    _collect_pair_annotation_track,
    _collect_region_tracks,
    _collect_sequence_diff_track,
    _sequence_diff_mask,
)
from .utils import (
    _add_panel_title,
    _color_track_rgba,
    _comparison_pair_specs,
    _default_bead_size,
    _draw_lw_glyphs,
    _fill_panel_axes,
    _maybe_add_color_track_legend,
    _pair_tier_color_map,
    _primary_pair_set,
    _region_colors,
    _resolve_bead_colors,
    _resolve_panel_axes,
    _resolve_show_bases,
    _structure_preflight,
)


def plot_circular_diagram(
    dot_bracket: str,
    sequence: str | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_bases: bool | None = None,
    show_ends: bool = True,
    tracks: Sequence[Track] | None = None,
    reference_dot_bracket: str | None = None,
    show_pseudoknot_tiers: bool = True,
    chord_curvature: float | str = "auto",
    pair_color: str = "#2563eb",
    mfe_pair_color: str | None = None,
    backbone_color: str = "#404040",
    palette: str = "nature",
    base_colors: Mapping[str, str] | None = None,
    bead_size: float | None = None,
    bead_edge_color: str = "#333333",
    base_font_size: float = 6.0,
) -> Figure:
    """
    Plot a dot-bracket RNA secondary structure as a circular diagram.

    Args:
        dot_bracket: Dot-bracket notation.
        sequence: RNA sequence. Bases are used for bead colors and optional labels. Use ``None`` when sequence
            information is not available.
        ax: Optional matplotlib axes.
        title: Plot title.
        show_bases: Whether to render base labels. Defaults to true for structures up to 120 nt.
        tracks: Optional sequence of overlay tracks. [BaseValueTrack][] and
            [BaseCategoryTrack][] render as outer-ring wedges; [RegionTrack][] renders as
            an outer-ring arc with label; [PairProbabilityTrack][] renders probability
            chords; [SequenceDiffTrack][] emphasizes changed base labels;
            [PairAnnotationTrack][] renders Leontis-Westhof glyphs.
        reference_dot_bracket: Optional reference dot-bracket. When provided, chords are
            colored as true positives, false positives, and false negatives.
        show_pseudoknot_tiers: When ``True``, pairs from non-primary brackets are colored from
            [PSEUDOKNOT_TIER_COLORS][].
        chord_curvature: Bézier curvature of the pair chords. ``"auto"`` adapts curvature to
            sequence length, pair density, and stacked stems to reduce local overlap. ``0.0``
            draws straight chords; ``1.0`` curves each chord through the origin (the bundled
            Circos-style look). Numeric values interpolate continuously.
        show_ends: When ``True``, draws small "5'" and "3'" labels just outside the backbone
            circle at the sequence start and end.
        pair_color: Chord color (only used when ``reference_dot_bracket`` is ``None``).
        mfe_pair_color: Color used for MFE chords when a [PairProbabilityTrack][] is also
            present, so the MFE structure stays visible without taking over the probability
            color scale. Defaults to a subtle blue-gray when BPP is shown; otherwise falls
            back to ``pair_color``.
        backbone_color: Backbone line color.
        palette: Name from [NUCLEOTIDE_PALETTES][].
        base_colors: Optional per-base color overrides.
        bead_size: Matplotlib scatter size for nucleotides. If not set, a length-aware default is used.
        bead_edge_color: Nucleotide bead edge color.
        base_font_size: Font size for base labels.

    Returns:
        The matplotlib figure containing the plot.

    Raises:
        ValueError: If ``dot_bracket`` is empty, ``sequence`` or ``reference_dot_bracket``
            length disagrees with ``dot_bracket``, or ``chord_curvature`` is neither
            ``"auto"`` nor a finite number in ``[0.0, 1.0]``.

    Examples:
        >>> from multimolecule.visualization.rna import plot_circular_diagram
        >>> fig = plot_circular_diagram("(((...)))", "GGGAAAUCC")
        >>> type(fig).__name__
        'Figure'
    """
    length, pairs = _structure_preflight(dot_bracket, sequence, reference_dot_bracket)
    colors = resolve_nucleotide_colors(palette, base_colors)
    if bead_size is None:
        bead_size = _default_bead_size(length)
    fig, ax, created_ax = _resolve_panel_axes(ax, figsize=(5.0, 5.0))

    theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, length, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)

    closed_xs = np.r_[xs, xs[0]]
    closed_ys = np.r_[ys, ys[0]]
    ax.plot(closed_xs, closed_ys, color=backbone_color, linewidth=0.8, zorder=1)

    probability_tracks = _collect_probability_tracks(tracks, length)
    curvature_for = _circular_chord_curvature_fn(chord_curvature, pairs, length)
    _circular_render_probability(ax, probability_tracks, theta, curvature_for)

    pair_tier_colors = _pair_tier_color_map(
        pairs,
        _primary_pair_set(dot_bracket, pairs),
        pair_color,
        show_pseudoknot_tiers,
    )
    effective_mfe_color = mfe_pair_color
    default_probability_mfe_overlay = False
    if effective_mfe_color is None and probability_tracks:
        effective_mfe_color = _DEFAULT_MFE_PROBABILITY_COLOR
        default_probability_mfe_overlay = True
    if reference_dot_bracket is None:
        if effective_mfe_color is None:
            specs = [(int(i), int(j), pair_tier_colors[(int(i), int(j))], 1.0, 0.78, "-") for i, j in pairs.tolist()]
            _circular_draw_chord_collection(ax, specs, theta, curvature_for, zorder=2)
        else:
            linewidth = _DEFAULT_MFE_PROBABILITY_LINEWIDTH if default_probability_mfe_overlay else 1.3
            alpha = _DEFAULT_MFE_PROBABILITY_ALPHA if default_probability_mfe_overlay else 0.92
            specs = [(int(i), int(j), effective_mfe_color, linewidth, alpha, "-") for i, j in pairs.tolist()]
            _circular_draw_chord_collection(ax, specs, theta, curvature_for, zorder=3)
    else:
        specs = _comparison_pair_specs(
            dot_bracket,
            reference_dot_bracket,
            fn_linewidth=0.85,
            tp_linewidth=1.1,
            fn_alpha=0.72,
            tp_alpha=0.92,
        )
        _circular_draw_chord_collection(ax, specs, theta, curvature_for, zorder=2)

    color_source = _collect_color_track(tracks, length)
    _circular_render_color_ring(ax, color_source, theta)
    region_tracks = _collect_region_tracks(tracks, length)
    _circular_render_regions(ax, region_tracks, theta)

    pair_annotation_track = _collect_pair_annotation_track(tracks, length, pairs)
    if pair_annotation_track is not None:
        _circular_render_pair_annotations(ax, pair_annotation_track, theta)

    bead_colors = _resolve_bead_colors(sequence, length, colors, color_source)
    diff_track = _collect_sequence_diff_track(tracks)
    diff_mask = _sequence_diff_mask(diff_track, sequence, length)
    if diff_track is not None and diff_mask.any():
        edge_colors = [diff_track.highlight_color if diff_mask[i] else bead_edge_color for i in range(length)]
        edge_widths = [1.2 if diff_mask[i] else 0.3 for i in range(length)]
        ax.scatter(xs, ys, s=bead_size, c=bead_colors, edgecolors=edge_colors, linewidths=edge_widths, zorder=3)
    else:
        ax.scatter(xs, ys, s=bead_size, c=bead_colors, edgecolors=bead_edge_color, linewidths=0.3, zorder=3)

    show_bases = _resolve_show_bases(sequence, show_bases, length, max_length=120)
    if show_bases:
        assert sequence is not None
        for index, base in enumerate(sequence[:length]):
            radius = 1.12
            color = diff_track.highlight_color if (diff_track is not None and diff_mask[index]) else "#000000"
            weight = "bold" if (diff_track is not None and diff_track.bold_label and diff_mask[index]) else "normal"
            ax.text(
                radius * xs[index],
                radius * ys[index],
                base.upper(),
                ha="center",
                va="center",
                fontsize=base_font_size,
                color=color,
                fontweight=weight,
                zorder=4,
            )

    if show_ends and length >= 2:
        _circular_render_end_labels(ax, theta)

    outer_extent = _circular_outer_extent(color_source, region_tracks, has_ends=show_ends)
    ax.set_xlim(-outer_extent, outer_extent)
    ax.set_ylim(-outer_extent, outer_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if created_ax:
        _fill_panel_axes(ax)
    if reference_dot_bracket is not None:
        _add_comparison_legend(ax)
    _maybe_add_color_track_legend(fig, ax, color_source)
    _maybe_add_probability_colorbar(fig, ax, probability_tracks)
    _add_panel_title(ax, title)
    return fig


_CIRC_RING_INNER = 1.02
_CIRC_RING_OUTER = 1.07
_CIRC_REGION_INNER = 1.09
_CIRC_REGION_OUTER = 1.14
_CIRC_REGION_LABEL = 1.20
_BEZIER_CHORD_POINTS = 48
ChordSpec = tuple[int, int, object, float, float, str]


def _bezier_chord(
    theta_i: float,
    theta_j: float,
    *,
    curvature: float = 1.0,
    n_points: int = _BEZIER_CHORD_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic Bézier chord between two positions on the unit circle.

    The control point is ``(1 - curvature) * midpoint(P0, P2)``: ``curvature=0`` produces a
    straight chord (P1 = midpoint) and ``curvature=1`` pulls the curve through the origin (the
    CS²BP² / Circos look). Interpolates continuously between the two.
    """
    p0 = np.array([np.cos(theta_i), np.sin(theta_i)])
    p2 = np.array([np.cos(theta_j), np.sin(theta_j)])
    midpoint = (p0 + p2) / 2.0
    p1 = (1.0 - float(curvature)) * midpoint
    t = np.linspace(0.0, 1.0, n_points)[:, None]
    points = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    return points[:, 0], points[:, 1]


def _circular_draw_chord_collection(
    ax: Axes,
    specs: Sequence[ChordSpec],
    theta: np.ndarray,
    curvature_for: Callable[[int, int], float],
    *,
    zorder: float,
) -> None:
    if not specs:
        return
    by_style: dict[str, tuple[list[np.ndarray], list[object], list[float]]] = {}
    for i, j, color, linewidth, alpha, linestyle in specs:
        cx, cy = _bezier_chord(theta[i], theta[j], curvature=curvature_for(i, j))
        segments, colors, linewidths = by_style.setdefault(linestyle, ([], [], []))
        segments.append(np.column_stack((cx, cy)))
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


def _circular_chord_curvature_fn(
    chord_curvature: float | str,
    pairs: np.ndarray,
    length: int,
) -> Callable[[int, int], float]:
    if isinstance(chord_curvature, str):
        if chord_curvature != "auto":
            raise ValueError("chord_curvature must be 'auto' or a number between 0.0 and 1.0.")
        return _adaptive_circular_chord_curvature_fn(pairs, length)

    value = float(chord_curvature)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("chord_curvature must be 'auto' or a number between 0.0 and 1.0.")
    return lambda _i, _j: value


def _adaptive_circular_chord_curvature_fn(
    pairs: np.ndarray,
    length: int,
) -> Callable[[int, int], float]:
    max_span = max(length // 2, 1)

    def curvature(i: int, j: int) -> float:
        span = _circular_pair_span(i, j, length)
        value = span / max_span
        return float(min(1.0, max(0.0, value)))

    return curvature


def _circular_pair_span(i: int, j: int, length: int) -> int:
    distance = abs(int(j) - int(i))
    return min(distance, length - distance)


def _circular_render_color_ring(
    ax: Axes,
    color_track: ColorTrack | None,
    theta: np.ndarray,
) -> None:
    if color_track is None:
        return
    rgba = _color_track_rgba(color_track)
    length = theta.size
    if length == 0:
        return
    wedge_half = np.pi / max(length, 1)
    polygons: list[np.ndarray] = []
    for i in range(length):
        phi = np.linspace(theta[i] - wedge_half, theta[i] + wedge_half, 12)
        inner_x = _CIRC_RING_INNER * np.cos(phi)
        inner_y = _CIRC_RING_INNER * np.sin(phi)
        outer_x = _CIRC_RING_OUTER * np.cos(phi[::-1])
        outer_y = _CIRC_RING_OUTER * np.sin(phi[::-1])
        polygons.append(np.column_stack((np.r_[inner_x, outer_x], np.r_[inner_y, outer_y])))
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors=rgba,
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )
    )


_CIRC_END_LABEL = 1.16
_CIRC_LW_GLYPH_INSET = 0.06


def _circular_render_pair_annotations(
    ax: Axes,
    track: PairAnnotationTrack,
    theta: np.ndarray,
) -> None:
    """Draw LW glyphs just inside the backbone circle at each annotated pair endpoint."""
    radius = 1.0 - _CIRC_LW_GLYPH_INSET
    glyphs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (i, j), annotation in track.annotations.items():
        theta_i = float(theta[i])
        theta_j = float(theta[j])
        glyphs.setdefault((annotation.edge_5p, annotation.orientation), []).append(
            (radius * np.cos(theta_i), radius * np.sin(theta_i))
        )
        glyphs.setdefault((annotation.edge_3p, annotation.orientation), []).append(
            (radius * np.cos(theta_j), radius * np.sin(theta_j))
        )
    for (edge, orientation), points in glyphs.items():
        _draw_lw_glyphs(
            ax,
            points=points,
            edge=edge,
            orientation=orientation,
            size=track.glyph_size,
            color=track.glyph_color,
        )


def _circular_render_end_labels(ax: Axes, theta: np.ndarray) -> None:
    """Draw small 5' / 3' annotations just outside the backbone circle."""
    nudge = min(0.35, max(0.18, 2 * np.pi / max(theta.size, 1) * 1.5))
    five_theta = float(theta[0]) + nudge
    three_theta = float(theta[-1]) - nudge
    ax.text(
        _CIRC_END_LABEL * np.cos(five_theta),
        _CIRC_END_LABEL * np.sin(five_theta),
        "5'",
        ha="right",
        va="center",
        fontsize=8,
        color="#333333",
        zorder=5,
    )
    ax.text(
        _CIRC_END_LABEL * np.cos(three_theta),
        _CIRC_END_LABEL * np.sin(three_theta),
        "3'",
        ha="left",
        va="center",
        fontsize=8,
        color="#333333",
        zorder=5,
    )


def _circular_render_regions(
    ax: Axes,
    region_tracks: list[RegionTrack],
    theta: np.ndarray,
) -> None:
    length = theta.size
    if length == 0:
        return
    offset = 0
    for track in region_tracks:
        colors = _region_colors(track, offset=offset)
        for (start, stop, label), color in zip(track.regions, colors):
            start_idx = max(0, int(start))
            stop_idx = min(int(stop), length - 1)
            if stop_idx < start_idx:
                continue
            phi = np.linspace(theta[start_idx], theta[stop_idx], max(8, stop_idx - start_idx + 1))
            inner_x = _CIRC_REGION_INNER * np.cos(phi)
            inner_y = _CIRC_REGION_INNER * np.sin(phi)
            outer_x = _CIRC_REGION_OUTER * np.cos(phi[::-1])
            outer_y = _CIRC_REGION_OUTER * np.sin(phi[::-1])
            ax.fill(
                np.r_[inner_x, outer_x],
                np.r_[inner_y, outer_y],
                color=color,
                alpha=min(1.0, track.alpha + 0.4),
                linewidth=0,
                zorder=3,
            )
            if track.show_labels and label:
                mid_theta = float((theta[start_idx] + theta[stop_idx]) / 2)
                ax.text(
                    _CIRC_REGION_LABEL * np.cos(mid_theta),
                    _CIRC_REGION_LABEL * np.sin(mid_theta),
                    label,
                    color=color,
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=5,
                )
        offset += len(track.regions)


def _circular_outer_extent(
    color_track: ColorTrack | None,
    region_tracks: list[RegionTrack],
    has_ends: bool = False,
) -> float:
    extent = 1.18
    if color_track is not None:
        extent = max(extent, _CIRC_RING_OUTER)
    has_region_label = any(t.show_labels and any(r[2] for r in t.regions) for t in region_tracks)
    if region_tracks:
        extent = max(extent, _CIRC_REGION_OUTER)
    if has_region_label:
        extent = max(extent, _CIRC_REGION_LABEL)
    if has_ends:
        extent = max(extent, _CIRC_END_LABEL)
    return extent


def _circular_render_probability(
    ax: Axes,
    probability_tracks: list[PairProbabilityTrack],
    theta: np.ndarray,
    curvature_for: Callable[[int, int], float],
) -> None:
    length = theta.size
    for track in probability_tracks:
        haze_specs: list[ChordSpec] = []
        for i, j, p in _probability_haze_pairs(track):
            if i >= length or j >= length:
                continue
            color, _ = _probability_color_and_width(track, p)
            haze_specs.append((i, j, color, _PROBABILITY_HAZE_LINEWIDTH, 0.12, "-"))
        _circular_draw_chord_collection(ax, haze_specs, theta, curvature_for, zorder=1.5)

        probability_specs: list[ChordSpec] = []
        for i, j, p in _probability_pairs(track):
            if i >= length or j >= length:
                continue
            color, width = _probability_color_and_width(track, p)
            probability_specs.append((i, j, color, width, _probability_alpha(p), "-"))
        _circular_draw_chord_collection(ax, probability_specs, theta, curvature_for, zorder=2)
