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

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from multimolecule.utils.rna.secondary_structure.notations import dot_bracket_to_pairs

from .utils import _add_panel_title, _create_panel_axes, _fill_panel_axes

_DEFAULT_STRUCTURE_COLORS: tuple[str, ...] = (
    "#0272b2",
    "#c93e3f",
    "#459434",
    "#a84e94",
    "#ec6f00",
    "#019aa3",
    "#cca02c",
    "#9c2826",
)
_CONSERVATION_GRADIENT: tuple[str, ...] = ("#f0f0f0", "#bfbfbf", "#6f7b91", "#0272b2")


def plot_alignment_arcs(
    aligned_dot_brackets: Sequence[str],
    aligned_sequences: Sequence[str],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    sequence_labels: Sequence[str] | None = None,
    structure_colors: Sequence[str] | None = None,
    show_conservation: bool = False,
    arc_height_scale: float = 1.0,
    base_font_size: float = 5.5,
    sequence_gap_chars: str = "-.~",
) -> Figure:
    """
    R-chie-style multi-structure alignment view.

    Renders one arc panel above stacked aligned sequence rows. Pairs from each input dot-bracket
    are drawn as arcs in a distinct color; aligned positions occupy the same x coordinate so
    structures can be compared column by column. Optionally tints sequence cells by per-column
    conservation.

    Args:
        aligned_dot_brackets: Aligned dot-bracket structures, same length as ``aligned_sequences``.
            Bracket pairs span alignment columns directly. ``-`` is treated as an
            alignment gap and rendered as an unpaired column; other invalid symbols still
            raise a parser error. Dot-bracket gaps must use ``-`` and ``.`` remains the
            normal unpaired symbol.
        aligned_sequences: Aligned RNA sequences. All must have the same length (alignment width)
            including gap characters.
        ax: Optional matplotlib axes. If provided, the alignment + arcs render into this single
            axes. If ``None``, a new figure with a length-aware size is created.
        title: Plot title.
        sequence_labels: Optional row labels (e.g. species). Defaults to ``"seq 1", "seq 2", ...``.
        structure_colors: Per-structure arc colors. Defaults to a discrete categorical palette.
        show_conservation: When ``True``, color each sequence cell by per-column identity
            conservation (light = variable, dark = conserved).
        arc_height_scale: Scales the height of arcs above the alignment baseline. 1.0 produces
            semicircles; smaller values flatten them.
        base_font_size: Font size for the per-cell base labels.
        sequence_gap_chars: Characters in ``aligned_sequences`` treated as gaps for conservation.

    Returns:
        The matplotlib figure containing the plot.

    Raises:
        ValueError: If ``aligned_dot_brackets`` is empty, the dot-bracket and sequence
            row counts disagree, rows do not share a common width, ``sequence_labels``
            has the wrong length, ``structure_colors`` is empty, or ``arc_height_scale``
            is not a positive number ``<= 2.0``.

    Examples:
        >>> from multimolecule.visualization.rna import plot_alignment_arcs
        >>> fig = plot_alignment_arcs(["(((...)))", "((.....))"], ["GGGAAAUCC", "GGGAAAUCC"])
        >>> type(fig).__name__
        'Figure'
    """
    if len(aligned_dot_brackets) == 0:
        raise ValueError("aligned_dot_brackets must not be empty.")
    if len(aligned_dot_brackets) != len(aligned_sequences):
        raise ValueError(
            f"aligned_dot_brackets and aligned_sequences must have the same row count "
            f"({len(aligned_dot_brackets)} vs {len(aligned_sequences)})."
        )
    width = len(aligned_dot_brackets[0])
    if width == 0:
        raise ValueError("aligned_dot_brackets must have non-zero width.")
    for dbn in aligned_dot_brackets:
        if len(dbn) != width:
            raise ValueError(f"all aligned_dot_brackets must have the same length; got {len(dbn)} != {width}.")
    for seq in aligned_sequences:
        if len(seq) != width:
            raise ValueError(
                f"all aligned_sequences must have the same length as dot-brackets; got {len(seq)} != {width}."
            )

    row_count = len(aligned_dot_brackets)
    labels = list(sequence_labels) if sequence_labels is not None else [f"seq {i + 1}" for i in range(row_count)]
    if len(labels) != row_count:
        raise ValueError(f"sequence_labels has {len(labels)} entries but {row_count} rows provided.")
    palette = list(structure_colors) if structure_colors is not None else list(_DEFAULT_STRUCTURE_COLORS)
    if not palette:
        raise ValueError("structure_colors must not be empty.")
    if len(palette) < row_count:
        palette = (palette * ((row_count // len(palette)) + 1))[:row_count]

    arc_height_scale = _check_height_scale(arc_height_scale)
    arc_pair_lists = [_aligned_dot_bracket_pairs(_normalize_aligned_dot_bracket(dbn)) for dbn in aligned_dot_brackets]

    if ax is None:
        fig_width = min(16.0, max(7.0, width / 16))
        fig_height = 2.4 + 0.25 * row_count
        fig, ax = _create_panel_axes(figsize=(fig_width, fig_height))
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    arc_top = _alignment_render_arcs(ax, arc_pair_lists, palette, arc_height_scale)
    _alignment_render_rows(
        ax,
        aligned_sequences=aligned_sequences,
        labels=labels,
        base_font_size=base_font_size,
        show_conservation=show_conservation,
        sequence_gap_chars=sequence_gap_chars,
    )

    ax.set_xlim(-1.5, width - 0.5)
    ax.set_ylim(-row_count - 0.5, arc_top + 0.2)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _alignment_add_structure_legend(ax, palette, row_count)
    if created_ax:
        _fill_panel_axes(ax)
    _add_panel_title(ax, title)
    return fig


def _normalize_aligned_dot_bracket(dot_bracket: str) -> str:
    return dot_bracket.replace("-", ".")


def _aligned_dot_bracket_pairs(dot_bracket: str) -> list[tuple[int, int]]:
    return [(int(i), int(j)) for i, j in dot_bracket_to_pairs(dot_bracket).tolist()]


def _alignment_render_arcs(
    ax: Axes,
    arc_pair_lists: Sequence[Sequence[tuple[int, int]]],
    palette: Sequence[str],
    height_scale: float,
) -> float:
    max_height = 0.0
    segments: list[np.ndarray] = []
    colors: list[str] = []
    theta = np.linspace(0.0, np.pi, 64)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    for structure_index, pairs in enumerate(arc_pair_lists):
        color = palette[structure_index]
        for i, j in pairs:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            center = (a + b) / 2.0
            radius = (b - a) / 2.0
            height = radius * height_scale
            xs = center + radius * cos_theta
            ys = height * sin_theta
            segments.append(np.column_stack((xs, ys)))
            colors.append(color)
            if height > max_height:
                max_height = height
    if segments:
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.9, alpha=0.78, zorder=2))
    baseline_y = 0.0
    ax.axhline(baseline_y, color="#404040", linewidth=0.6, zorder=1)
    return max(max_height, 0.1)


def _alignment_render_rows(
    ax: Axes,
    *,
    aligned_sequences: Sequence[str],
    labels: Sequence[str],
    base_font_size: float,
    show_conservation: bool,
    sequence_gap_chars: str,
) -> None:
    gap_set = set(sequence_gap_chars)
    if show_conservation:
        column_conservation = _column_conservation(aligned_sequences, gap_set)
    else:
        column_conservation = None
    for row_index, sequence in enumerate(aligned_sequences):
        y = -row_index - 0.5
        ax.text(-1.0, y, labels[row_index], ha="right", va="center", fontsize=base_font_size + 1, color="#333333")
        for column, base in enumerate(sequence):
            base_upper = base.upper() if base not in gap_set else "-"
            background = _conservation_color(column_conservation[column]) if column_conservation is not None else None
            if background is not None and base not in gap_set:
                ax.add_patch(_alignment_cell_patch(column, y, background))
            ax.text(
                column,
                y,
                base_upper,
                ha="center",
                va="center",
                fontsize=base_font_size,
                color="#222222" if base not in gap_set else "#9aa0a6",
            )


def _alignment_cell_patch(column: int, y: float, color: str):
    return Rectangle((column - 0.45, y - 0.4), 0.9, 0.8, facecolor=color, edgecolor="none", zorder=0)


def _alignment_add_structure_legend(ax: Axes, palette: Sequence[str], row_count: int) -> None:
    handles = [Line2D([0], [0], color=palette[i], linewidth=1.2, label=f"structure {i + 1}") for i in range(row_count)]
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7, handlelength=1.6)


def _column_conservation(aligned_sequences: Sequence[str], gap_set: set[str]) -> np.ndarray:
    width = len(aligned_sequences[0])
    conservation = np.zeros(width, dtype=float)
    for column in range(width):
        counts: dict[str, int] = {}
        non_gap_total = 0
        for sequence in aligned_sequences:
            base = sequence[column]
            if base in gap_set:
                continue
            non_gap_total += 1
            key = base.upper()
            counts[key] = counts.get(key, 0) + 1
        if non_gap_total == 0:
            conservation[column] = 0.0
            continue
        conservation[column] = max(counts.values()) / non_gap_total
    return conservation


def _conservation_color(score: float) -> str:
    score = float(min(1.0, max(0.0, score)))
    bins = len(_CONSERVATION_GRADIENT) - 1
    index = int(round(score * bins))
    return _CONSERVATION_GRADIENT[index]


def _check_height_scale(value: float) -> float:
    height = float(value)
    if not np.isfinite(height) or height <= 0.0 or height > 2.0:
        raise ValueError("arc_height_scale must be a positive number <= 2.0.")
    return height
