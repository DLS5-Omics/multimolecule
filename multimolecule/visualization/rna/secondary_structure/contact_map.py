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

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..palettes import PAIR_PROBABILITY_CMAP
from .tracks import (
    ColorTrack,
    RegionTrack,
    SequenceDiffTrack,
    Track,
    _collect_color_tracks,
    _collect_region_tracks,
    _collect_sequence_diff_track,
    _sequence_diff_mask,
)
from .utils import (
    _add_attached_colorbar,
    _add_panel_title,
    _check_sequence_length,
    _color_track_rgba,
    _create_panel_axes,
    _fill_panel_axes,
    _region_colors,
)


def plot_contact_map(
    contact_map: Any,
    sequence: str | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = PAIR_PROBABILITY_CMAP,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    colorbar: bool = False,
    tracks: Sequence[Track] | None = None,
    reference: Any | None = None,
) -> Figure:
    """
    Plot a contact map or contact-probability matrix.

    Args:
        contact_map: 2D array-like contact map (or contact-probability matrix).
        sequence: Optional sequence aligned to the contact-map axes. Required when using
            [SequenceDiffTrack][].
        ax: Optional matplotlib axes.
        title: Plot title.
        cmap: Matplotlib colormap name.
        vmin: Lower color scale bound.
        vmax: Upper color scale bound.
        colorbar: Whether to add a colorbar.
        tracks: Optional sequence of overlay tracks. [BaseValueTrack][] /
            [BaseCategoryTrack][] render as sidebars; [RegionTrack][] renders as
            diagonal boxes; [SequenceDiffTrack][] emphasizes changed tick labels.
            [PairProbabilityTrack][] is a no-op here because the contact map already
            shows the probability matrix directly. [PairAnnotationTrack][] is also a
            no-op on contact maps.
        reference: Optional reference contact map (2-D array-like) or dot-bracket string.
            When provided, the upper triangle of the rendered map shows ``contact_map`` and the
            lower triangle shows the reference, matching the ViennaRNA dot-plot convention.
            Named differently from the other plot functions' ``reference_dot_bracket`` because
            this parameter also accepts a 2-D array.

    Returns:
        The matplotlib figure containing the plot.

    Raises:
        ValueError: If ``contact_map`` is not a square 2-D array, ``sequence`` length
            disagrees with the matrix, or ``reference`` (when provided) has a mismatched
            shape or dot-bracket length.

    Examples:
        >>> import numpy as np
        >>> from multimolecule.visualization.rna import plot_contact_map
        >>> fig = plot_contact_map(np.eye(9))
        >>> type(fig).__name__
        'Figure'
    """
    matrix = np.asarray(contact_map)
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        matrix = matrix.squeeze(-1)
    if matrix.ndim != 2:
        raise ValueError(f"contact_map must be 2D, but got shape {matrix.shape}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"contact_map must be square, but got shape {matrix.shape}.")
    length = matrix.shape[0]
    _check_sequence_length(sequence, length, target="contact_map")

    display = matrix.astype(float, copy=True)
    if reference is not None:
        ref_matrix = _coerce_reference_matrix(reference, target_shape=matrix.shape)
        display = _combine_dual_triangle(display, ref_matrix)

    if ax is None:
        fig, ax = _create_panel_axes(figsize=(5.0, 5.0))
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    image = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        _add_attached_colorbar(fig, ax, image)

    if reference is not None:
        _contact_map_dual_triangle_decoration(ax, length=length)

    if tracks:
        _contact_map_render_regions(ax, _collect_region_tracks(tracks, length))
        _contact_map_render_color_sidebars(ax, _collect_color_tracks(tracks, length))
        _contact_map_render_sequence_diff(ax, _collect_sequence_diff_track(tracks), sequence, length)

    if created_ax:
        _fill_panel_axes(ax)
    _add_panel_title(ax, title)
    return fig


def _coerce_reference_matrix(reference, target_shape: tuple[int, int]) -> np.ndarray:
    """Accept a 2-D array-like or a dot-bracket string and return a float matrix shaped ``target_shape``."""
    from multimolecule.utils.rna.secondary_structure import dot_bracket_to_contact_map

    if isinstance(reference, str):
        if len(reference) != target_shape[0]:
            raise ValueError(f"reference dot-bracket length {len(reference)} != contact_map size {target_shape[0]}.")
        ref = np.asarray(dot_bracket_to_contact_map(reference), dtype=float)
    else:
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 3 and ref.shape[-1] == 1:
            ref = ref.squeeze(-1)
    if ref.shape != target_shape:
        raise ValueError(f"reference shape {ref.shape} != contact_map shape {target_shape}.")
    return ref


def _combine_dual_triangle(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Combine into a single display matrix: upper triangle = predicted, lower triangle = reference."""
    length = predicted.shape[0]
    out = predicted.copy()
    lower = np.tril_indices(length, k=-1)
    out[lower] = reference[lower]
    return out


def _contact_map_dual_triangle_decoration(ax: Axes, length: int) -> None:
    """Draw the diagonal separator and corner labels for the dual-triangle contact map."""
    ax.plot([-0.5, length - 0.5], [-0.5, length - 0.5], color="#444444", linewidth=0.7, alpha=0.7, zorder=2)
    ax.text(
        length - 1,
        0,
        "predicted",
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
        zorder=4,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
    )
    ax.text(
        0,
        length - 1,
        "reference",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
        zorder=4,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
    )


def _contact_map_render_regions(ax: Axes, region_tracks: list[RegionTrack]) -> None:
    offset = 0
    for track in region_tracks:
        colors = _region_colors(track, offset=offset)
        for (start, stop, label), color in zip(track.regions, colors):
            start_idx = int(start)
            stop_idx = int(stop)
            if stop_idx < start_idx:
                continue
            width = stop_idx - start_idx + 1
            rect = Rectangle(
                (start_idx - 0.5, start_idx - 0.5),
                width,
                width,
                edgecolor=color,
                facecolor="none",
                linewidth=1.4,
                zorder=3,
            )
            ax.add_patch(rect)
            if track.show_labels and label:
                ax.text(
                    start_idx + width / 2 - 0.5,
                    start_idx - 0.45,
                    label,
                    color=color,
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    zorder=4,
                )
        offset += len(track.regions)


def _contact_map_render_sequence_diff(
    ax: Axes,
    track: SequenceDiffTrack | None,
    sequence: str | None,
    length: int,
) -> None:
    diff_mask = _sequence_diff_mask(track, sequence, length)
    if track is None or not diff_mask.any():
        return
    diff_positions = {int(i) for i in np.flatnonzero(diff_mask)}
    _highlight_contact_map_ticks(ax, axis="x", positions=diff_positions, length=length, track=track)
    _highlight_contact_map_ticks(ax, axis="y", positions=diff_positions, length=length, track=track)


def _highlight_contact_map_ticks(
    ax: Axes,
    *,
    axis: str,
    positions: set[int],
    length: int,
    track: SequenceDiffTrack,
) -> None:
    ticks = _merge_integer_ticks(ax.get_xticks() if axis == "x" else ax.get_yticks(), positions, length)
    labels = [str(tick) for tick in ticks]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", direction="in", pad=-10, length=2)
        tick_labels = ax.get_xticklabels()
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", direction="in", pad=-14, length=2)
        tick_labels = ax.get_yticklabels()
    for label in tick_labels:
        text = label.get_text()
        if not text:
            continue
        position = int(text)
        if position in positions:
            label.set_color(track.highlight_color)
            if track.bold_label:
                label.set_fontweight("bold")


def _merge_integer_ticks(ticks: Sequence[float], positions: set[int], length: int) -> list[int]:
    merged: set[int] = set(positions)
    for tick in ticks:
        value = float(tick)
        nearest = round(value)
        if abs(value - nearest) <= 1e-6 and 0 <= nearest < length:
            merged.add(int(nearest))
    return sorted(merged)


def _contact_map_render_color_sidebars(
    ax: Axes,
    color_tracks: list[ColorTrack],
) -> None:
    if not color_tracks:
        return

    size = min(0.04, 0.25 / len(color_tracks))
    # Sidebars MUST span the full axes width (and height for the left bar) so column i in
    # the sidebar sits directly above column i in the matrix. If a colorbar is also present
    # it occupies the right edge of the axes; the sidebar's zorder=6 paints over its top so
    # the small overlap stays visually clean rather than the sidebar being horizontally
    # compressed and silently mis-registering its columns with the data.
    for index, track in enumerate(color_tracks):
        rgba = _color_track_rgba(track)
        rgba_row = rgba[np.newaxis, :, :]
        rgba_col = rgba[:, np.newaxis, :]

        top_ax = ax.inset_axes([0.0, 1.0 - (index + 1) * size, 1.0, size], transform=ax.transAxes, zorder=6)
        top_ax.imshow(rgba_row, aspect="auto", interpolation="nearest")
        top_ax.set_xticks([])
        top_ax.set_yticks([])
        top_ax.set_frame_on(False)

        left_ax = ax.inset_axes([index * size, 0.0, size, 1.0], transform=ax.transAxes, zorder=6)
        left_ax.imshow(rgba_col, aspect="auto", interpolation="nearest")
        left_ax.set_xticks([])
        left_ax.set_yticks([])
        left_ax.set_frame_on(False)
