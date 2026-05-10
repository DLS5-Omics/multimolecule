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

from typing import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from multimolecule.utils.rna.secondary_structure import noncanonical_pairs_set

from ..palettes import resolve_nucleotide_colors
from .comparison import _add_comparison_legend
from .layout import _normalize_coords, _primary_layout_pairs, _rotate_coords, _tree_layout
from .probability import (
    _PROBABILITY_HAZE_LINEWIDTH,
    _collect_probability_tracks,
    _maybe_add_probability_colorbar,
    _probability_alpha,
    _probability_color_and_width,
    _probability_haze_pairs,
    _probability_pairs,
)
from .tracks import (
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
    _comparison_pair_specs,
    _default_bead_size,
    _default_structure_figsize,
    _draw_lw_glyphs,
    _fill_panel_axes,
    _maybe_add_color_track_legend,
    _pair_tier_color_map,
    _region_colors,
    _resolve_bead_colors,
    _resolve_panel_axes,
    _resolve_show_bases,
    _structure_preflight,
)


def plot_planar_graph(
    dot_bracket: str,
    sequence: str | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    bead_size: float | None = None,
    show_bases: bool | None = None,
    reference_dot_bracket: str | None = None,
    tracks: Sequence[Track] | None = None,
    show_pseudoknot_tiers: bool = True,
    palette: str = "nature",
    base_colors: Mapping[str, str] | None = None,
    backbone_color: str = "#cccccc",
    pair_color: str = "#333333",
    bead_edge_color: str = "#333333",
    rotate: float = 0.0,
    show_noncanonical_pairs: bool = False,
    noncanonical_marker_color: str = "#cc0000",
    g4_motifs: Sequence[Sequence[Sequence[int]]] | None = None,
    g4_color: str = "#9c7717",
) -> Figure:
    """
    Plot an RNA secondary structure as a planar graph.

    ``matplotlib`` is required for plotting. The default layout is implemented in MultiMolecule and supports
    every dot-bracket pair type parsed by [dot_bracket_to_pairs][].

    Args:
        dot_bracket: Dot-bracket notation.
        sequence: RNA sequence. Bases are used for bead colors and optional labels. Use ``None`` when sequence
            information is not available.
        ax: Optional matplotlib axes.
        title: Plot title.
        bead_size: Matplotlib scatter size for nucleotides. If not set, a length-aware default is used.
        show_bases: Whether to render base labels. Defaults to true for structures up to 160 nt.
        reference_dot_bracket: Optional reference dot-bracket notation. When provided, pair lines are colored as
            true positives, false positives, and false negatives.
        tracks: Optional sequence of overlay tracks. [BaseValueTrack][] and
            [BaseCategoryTrack][] color the nucleotide beads; [RegionTrack][] highlights
            backbone spans and renders labels; [PairProbabilityTrack][] renders BPP haze
            beneath the MFE pair lines; [SequenceDiffTrack][] emphasizes positions where
            the rendered sequence differs from a reference; [PairAnnotationTrack][] renders
            Leontis-Westhof glyphs near each annotated pair's endpoints.
        show_pseudoknot_tiers: When ``True``, pairs from non-primary brackets are colored from
            [PSEUDOKNOT_TIER_COLORS][] so crossing tiers are visually distinguishable. When
            ``False`` every pair uses ``pair_color`` regardless of nesting tier.
        palette: Name from [NUCLEOTIDE_PALETTES][].
        base_colors: Optional per-base color overrides.
        backbone_color: Backbone line color.
        pair_color: Base-pair line color.
        bead_edge_color: Nucleotide bead edge color.
        rotate: Counter-clockwise layout rotation in degrees.
        show_noncanonical_pairs: When ``True``, mark pairs whose base composition is not
            Watson-Crick or wobble with a small open square at the
            pair midpoint. This is a Leontis-Westhof-style placeholder — we detect "this pair
            is non-canonical" but do not yet annotate the edge type (WC / Hoogsteen / Sugar).
        noncanonical_marker_color: Edge color for the non-canonical pair marker.
        g4_motifs: Optional list of G-quadruplex motifs. Each motif is a sequence of four
            G-tracts and each G-tract is a sequence of position indices. Detection of G4
            motifs is the caller's responsibility; this renders the given motifs as
            highlighted G beads connected by a dashed tetrad polygon through the G-tract
            centroids.
        g4_color: Color for G-quadruplex highlighting.

    Returns:
        The matplotlib figure containing the plot.

    Raises:
        ValueError: If ``dot_bracket`` is empty, ``sequence`` or ``reference_dot_bracket``
            length disagrees with ``dot_bracket``, ``show_noncanonical_pairs=True`` is
            passed without a ``sequence``, or any ``g4_motifs`` entry is malformed.

    Examples:
        >>> from multimolecule.visualization.rna import plot_planar_graph
        >>> fig = plot_planar_graph("(((...)))", "GGGAAAUCC")
        >>> type(fig).__name__
        'Figure'
    """
    length, pairs = _structure_preflight(dot_bracket, sequence, reference_dot_bracket)
    if show_noncanonical_pairs and sequence is None:
        raise ValueError("sequence is required when show_noncanonical_pairs=True.")
    g4_motif_list = _normalize_g4_motifs(g4_motifs, length)
    if bead_size is None:
        bead_size = _default_bead_size(length)
    palette_colors = resolve_nucleotide_colors(palette, base_colors)
    fig_size = _default_structure_figsize(length)
    fig, ax, created_ax = _resolve_panel_axes(ax, figsize=(fig_size, fig_size))

    layout_pairs = _primary_layout_pairs(dot_bracket, pairs)
    layout_pair_set = {tuple(pair) for pair in layout_pairs.tolist()}
    xs, ys = _tree_layout(length, layout_pairs)
    xs, ys = _rotate_coords(xs, ys, rotate)
    xs, ys = _normalize_coords(xs, ys)

    ax.plot(xs, ys, "-", color=backbone_color, linewidth=0.7, zorder=1)
    probability_tracks = _collect_probability_tracks(tracks, length)
    _planar_render_probability(ax, probability_tracks, xs, ys)
    pair_tier_colors = _pair_tier_color_map(pairs, layout_pair_set, pair_color, show_pseudoknot_tiers)
    if reference_dot_bracket is None:
        specs = []
        for i, j in pairs.tolist():
            key = (int(i), int(j))
            specs.append(
                (
                    key[0],
                    key[1],
                    pair_tier_colors[key],
                    0.9,
                    1.0 if key in layout_pair_set else 0.85,
                    "-" if key in layout_pair_set else "--",
                )
            )
        _planar_draw_pair_segments(ax, specs, xs, ys, zorder=2)
    else:
        specs = _comparison_pair_specs(
            dot_bracket,
            reference_dot_bracket,
            fn_linewidth=0.75,
            tp_linewidth=0.95,
            fn_alpha=0.72,
            tp_alpha=1.0,
        )
        _planar_draw_pair_segments(ax, specs, xs, ys, zorder=2)

    if show_noncanonical_pairs and pairs.size:
        assert sequence is not None
        _planar_render_noncanonical_markers(ax, pairs, sequence, xs, ys, noncanonical_marker_color)

    pair_annotation_track = _collect_pair_annotation_track(tracks, length, pairs)
    if pair_annotation_track is not None:
        _planar_render_pair_annotations(ax, pair_annotation_track, xs, ys)

    if g4_motif_list:
        _planar_render_g4_motifs(ax, g4_motif_list, xs, ys, g4_color)

    region_tracks = _collect_region_tracks(tracks, length)
    _planar_render_region_backbones(ax, region_tracks, xs, ys, alpha_overlay=True)

    color_source = _collect_color_track(tracks, length)
    bead_colors = _resolve_bead_colors(sequence, length, palette_colors, color_source)
    diff_track = _collect_sequence_diff_track(tracks)
    diff_mask = _sequence_diff_mask(diff_track, sequence, length)
    if diff_track is not None and diff_mask.any():
        edge_colors = [diff_track.highlight_color if diff_mask[i] else bead_edge_color for i in range(length)]
        edge_widths = [1.2 if diff_mask[i] else 0.3 for i in range(length)]
        ax.scatter(xs, ys, s=bead_size, c=bead_colors, edgecolors=edge_colors, linewidths=edge_widths, zorder=3)
    else:
        ax.scatter(xs, ys, s=bead_size, c=bead_colors, edgecolors=bead_edge_color, linewidths=0.3, zorder=3)

    show_bases = _resolve_show_bases(sequence, show_bases, length, max_length=160)
    if show_bases:
        assert sequence is not None
        for index, base in enumerate(sequence[:length]):
            color = diff_track.highlight_color if (diff_track is not None and diff_mask[index]) else "#000000"
            weight = "bold" if (diff_track is not None and diff_track.bold_label and diff_mask[index]) else "normal"
            ax.text(
                xs[index],
                ys[index],
                base.upper(),
                ha="center",
                va="center",
                fontsize=5,
                color=color,
                fontweight=weight,
                zorder=4,
            )

    _planar_render_region_labels(ax, region_tracks, xs, ys)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    span = max(x_max - x_min, y_max - y_min, 1.0)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    ax.set_xlim(center_x - span / 2, center_x + span / 2)
    ax.set_ylim(center_y - span / 2, center_y + span / 2)
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


PairSegmentSpec = tuple[int, int, object, float, float, str]


def _planar_draw_pair_segments(
    ax: Axes,
    specs: Sequence[PairSegmentSpec],
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    zorder: float,
) -> None:
    if not specs:
        return
    by_style: dict[str, tuple[list[np.ndarray], list[object], list[float]]] = {}
    for i, j, color, linewidth, alpha, linestyle in specs:
        segments, colors, linewidths = by_style.setdefault(linestyle, ([], [], []))
        segments.append(np.asarray([[xs[i], ys[i]], [xs[j], ys[j]]], dtype=float))
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


def _normalize_g4_motifs(
    g4_motifs: Sequence[Sequence[Sequence[int]]] | None,
    length: int,
) -> list[list[list[int]]]:
    if not g4_motifs:
        return []
    normalized: list[list[list[int]]] = []
    for motif in g4_motifs:
        if len(motif) != 4:
            raise ValueError(f"each G-quadruplex motif must have exactly 4 G-tracts, got {len(motif)}.")
        normalized_motif: list[list[int]] = []
        for tract in motif:
            indices = [int(idx) for idx in tract]
            if not indices:
                raise ValueError("G-quadruplex tract must not be empty.")
            for idx in indices:
                if idx < 0 or idx >= length:
                    raise ValueError(f"G-quadruplex index {idx} is outside structure length {length}.")
            normalized_motif.append(indices)
        normalized.append(normalized_motif)
    return normalized


def _planar_render_g4_motifs(
    ax: Axes,
    g4_motifs: Sequence[Sequence[Sequence[int]]],
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
) -> None:
    """Highlight G-quadruplex motifs: emphasize the four G-tracts and draw the tetrad polygon."""
    for motif in g4_motifs:
        tract_centroids: list[tuple[float, float]] = []
        for tract in motif:
            indices = list(tract)
            ax.scatter(
                xs[indices],
                ys[indices],
                s=42,
                marker="o",
                facecolors=color,
                edgecolors="#333333",
                linewidths=0.5,
                alpha=0.85,
                zorder=3.5,
            )
            tract_centroids.append((float(xs[indices].mean()), float(ys[indices].mean())))
        polygon = np.asarray(tract_centroids + [tract_centroids[0]], dtype=float)
        ax.plot(polygon[:, 0], polygon[:, 1], linestyle="--", color=color, linewidth=1.0, alpha=0.7, zorder=2.5)


def _planar_render_pair_annotations(
    ax: Axes,
    track: PairAnnotationTrack,
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """Draw LW glyphs near each annotated pair's endpoints (~1/4 and 3/4 along the line)."""
    glyphs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (i, j), annotation in track.annotations.items():
        glyphs.setdefault((annotation.edge_5p, annotation.orientation), []).append(
            (float(xs[i] * 0.75 + xs[j] * 0.25), float(ys[i] * 0.75 + ys[j] * 0.25))
        )
        glyphs.setdefault((annotation.edge_3p, annotation.orientation), []).append(
            (float(xs[i] * 0.25 + xs[j] * 0.75), float(ys[i] * 0.25 + ys[j] * 0.75))
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


def _planar_render_noncanonical_markers(
    ax: Axes,
    pairs: np.ndarray,
    sequence: str,
    xs: np.ndarray,
    ys: np.ndarray,
    marker_color: str,
) -> None:
    nc_pairs = noncanonical_pairs_set(pairs, sequence, unsafe=True)
    if not nc_pairs:
        return
    nc_array = np.asarray(sorted(nc_pairs), dtype=int)
    mid_x = (xs[nc_array[:, 0]] + xs[nc_array[:, 1]]) / 2.0
    mid_y = (ys[nc_array[:, 0]] + ys[nc_array[:, 1]]) / 2.0
    ax.scatter(
        mid_x,
        mid_y,
        s=22,
        marker="s",
        facecolors="white",
        edgecolors=marker_color,
        linewidths=0.9,
        zorder=4,
    )


def _planar_render_region_backbones(
    ax: Axes,
    region_tracks: list[RegionTrack],
    xs: np.ndarray,
    ys: np.ndarray,
    alpha_overlay: bool,
) -> None:
    offset = 0
    for track in region_tracks:
        colors = _region_colors(track, offset=offset)
        for (start, stop, _label), color in zip(track.regions, colors):
            start_idx = max(0, int(start))
            stop_idx = min(int(stop), xs.size - 1)
            if stop_idx < start_idx:
                continue
            ax.plot(
                xs[start_idx : stop_idx + 1],
                ys[start_idx : stop_idx + 1],
                "-",
                color=color,
                linewidth=2.0,
                alpha=track.alpha if alpha_overlay else 1.0,
                solid_capstyle="round",
                zorder=2.5,
            )
        offset += len(track.regions)


def _planar_render_region_labels(
    ax: Axes,
    region_tracks: list[RegionTrack],
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    offset = 0
    for track in region_tracks:
        if not track.show_labels:
            offset += len(track.regions)
            continue
        colors = _region_colors(track, offset=offset)
        for (start, stop, label), color in zip(track.regions, colors):
            if not label:
                continue
            start_idx = max(0, int(start))
            stop_idx = min(int(stop), xs.size - 1)
            if stop_idx < start_idx:
                continue
            mid_x = float(xs[start_idx : stop_idx + 1].mean())
            mid_y = float(ys[start_idx : stop_idx + 1].mean())
            ax.text(
                mid_x,
                mid_y,
                label,
                color=color,
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": color, "linewidth": 0.5},
            )
        offset += len(track.regions)


def _planar_render_probability(
    ax: Axes,
    probability_tracks: list[PairProbabilityTrack],
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    for track in probability_tracks:
        haze_specs: list[PairSegmentSpec] = []
        for i, j, p in _probability_haze_pairs(track):
            if i >= xs.size or j >= xs.size:
                continue
            color, _ = _probability_color_and_width(track, p)
            haze_specs.append((i, j, color, _PROBABILITY_HAZE_LINEWIDTH, 0.12, "-"))
        _planar_draw_pair_segments(ax, haze_specs, xs, ys, zorder=1.3)

        probability_specs: list[PairSegmentSpec] = []
        for i, j, p in _probability_pairs(track):
            if i >= xs.size or j >= xs.size:
                continue
            color, width = _probability_color_and_width(track, p)
            probability_specs.append((i, j, color, width, _probability_alpha(p), "-"))
        _planar_draw_pair_segments(ax, probability_specs, xs, ys, zorder=1.5)
