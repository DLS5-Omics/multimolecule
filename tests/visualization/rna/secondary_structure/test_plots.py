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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from multimolecule.visualization.rna import (
    PAIR_COMPARISON_COLORS,
    PSEUDOKNOT_TIER_COLORS,
    plot_alignment_arcs,
    plot_arc_diagram,
    plot_circular_diagram,
    plot_contact_map,
    plot_planar_graph,
)

STRUCTURE_PLOTS = [
    pytest.param(plot_planar_graph, id="planar"),
    pytest.param(plot_arc_diagram, id="arc"),
    pytest.param(plot_circular_diagram, id="circular"),
]


def _line_collections(ax):
    return [collection for collection in ax.collections if isinstance(collection, LineCollection)]


def _line_collection_segment_colors(ax) -> dict[tuple[int, int], str]:
    colors: dict[tuple[int, int], str] = {}
    for collection in _line_collections(ax):
        collection_colors = collection.get_colors()
        for segment, color in zip(collection.get_segments(), collection_colors):
            key = (round(float(segment[:, 0].min())), round(float(segment[:, 0].max())))
            colors[key] = to_hex(color)
    return colors


# ---------------------------------------------------------------------------
# Basic API: structure plots accept (dot_bracket[, sequence]) and return a Figure.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_structure_plot_accepts_sequence(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    fig = plot_fn(dbn, seq)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_structure_plot_accepts_missing_sequence(plot_fn, nested_structure) -> None:
    _, dbn = nested_structure
    fig = plot_fn(dbn)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_structure_plot_rejects_sequence_length_mismatch(plot_fn, nested_structure) -> None:
    _, dbn = nested_structure
    with pytest.raises(ValueError, match="sequence length"):
        plot_fn(dbn, "ACGU")


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_structure_plot_rejects_reference_length_mismatch(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    with pytest.raises(ValueError, match="reference_dot_bracket length"):
        plot_fn(dbn, seq, reference_dot_bracket=dbn + ".")


def test_plot_planar_graph_rejects_empty_dot_bracket() -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_planar_graph("", "")


def test_plot_contact_map_returns_figure(random_contact_map) -> None:
    fig = plot_contact_map(random_contact_map)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_contact_map_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="contact_map must be 2D"):
        plot_contact_map(np.zeros((4, 4, 4)))


def test_plot_contact_map_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="contact_map must be square"):
        plot_contact_map(np.zeros((3, 4)))


def test_plot_contact_map_rejects_sequence_length_mismatch(random_contact_map) -> None:
    with pytest.raises(ValueError, match="sequence length"):
        plot_contact_map(random_contact_map, sequence="ACGU")


# ---------------------------------------------------------------------------
# Titles render inside the panel — readers expect title text on the axes,
# not above it. We don't lock the exact (x, y) position because that's a
# stylistic detail; only that a title is present as an in-axes text artist.
# ---------------------------------------------------------------------------


def test_plot_titles_render_as_in_axes_text(nested_structure, random_contact_map) -> None:
    seq, dbn = nested_structure
    figures = [
        plot_planar_graph(dbn, seq, title="Panel title"),
        plot_arc_diagram(dbn, seq, title="Panel title"),
        plot_circular_diagram(dbn, seq, title="Panel title"),
        plot_contact_map(random_contact_map, title="Panel title"),
        plot_alignment_arcs(["...."], ["AAAA"], title="Panel title"),
    ]
    for fig in figures:
        ax = fig.axes[0]
        assert ax.get_title() == ""
        titles = [text for text in ax.texts if text.get_text() == "Panel title"]
        assert len(titles) == 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# G-quadruplex motifs (planar only).
# ---------------------------------------------------------------------------


def test_plot_planar_graph_g4_motifs_render() -> None:
    sequence = "GGGTTGGGTTGGGTTGGG"
    dot_bracket = "." * len(sequence)
    motif = [[0, 1, 2], [5, 6, 7], [10, 11, 12], [15, 16, 17]]
    fig = plot_planar_graph(dot_bracket, sequence, g4_motifs=[motif])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_planar_graph_g4_motifs_reject_wrong_tract_count() -> None:
    with pytest.raises(ValueError, match="exactly 4 G-tracts"):
        plot_planar_graph("." * 7, "GGGAGGG", g4_motifs=[[[0, 1, 2], [4, 5, 6]]])


def test_plot_planar_graph_g4_motifs_reject_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="outside structure length"):
        plot_planar_graph("." * 7, "GGGAGGG", g4_motifs=[[[0, 1, 99], [2], [3], [4]]])


# ---------------------------------------------------------------------------
# Non-canonical pair marker (planar only).
# ---------------------------------------------------------------------------


def test_plot_planar_graph_marks_noncanonical_pairs() -> None:
    # (0, 4) C-A is non-canonical; (1, 3) G-C is canonical.
    fig = plot_planar_graph("((.))", "CGGCA", show_noncanonical_pairs=True)
    plt.close(fig)


def test_plot_planar_graph_rejects_noncanonical_markers_without_sequence(nested_structure) -> None:
    _, dbn = nested_structure
    with pytest.raises(ValueError, match="sequence is required"):
        plot_planar_graph(dbn, show_noncanonical_pairs=True)


# ---------------------------------------------------------------------------
# Alignment arcs.
# ---------------------------------------------------------------------------


def test_plot_alignment_arcs_returns_figure() -> None:
    fig = plot_alignment_arcs(["(((....)))", "((......))"], ["GGGAAAUCCC", "GGGAAAUCCC"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_alignment_arcs_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_alignment_arcs(["....", "..."], ["AAAA", "AAA"])


def test_plot_alignment_arcs_rejects_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="same row count"):
        plot_alignment_arcs(["....", "..."], ["AAAA"])


def test_plot_alignment_arcs_with_gaps_and_conservation() -> None:
    sequences = ["GGGAAUCCC", "GG-AAUCCC", "GGGA-UCCC"]
    structures = ["(((...)))", "((....)).", "(((..)).)"]
    fig = plot_alignment_arcs(structures, sequences, show_conservation=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_alignment_arcs_accepts_dot_bracket_gap_columns() -> None:
    fig = plot_alignment_arcs(["(-)", "(.)"], ["AAA", "AAA"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_alignment_arcs_rejects_invalid_dot_bracket_symbols() -> None:
    with pytest.raises(ValueError, match="Invalid symbol"):
        plot_alignment_arcs(["(?)"], ["AAA"])


def test_plot_alignment_arcs_rejects_bad_height_scale() -> None:
    with pytest.raises(ValueError, match="arc_height_scale"):
        plot_alignment_arcs(["...."], ["AAAA"], arc_height_scale=0.0)


def test_plot_alignment_arcs_rejects_empty_structure_colors() -> None:
    with pytest.raises(ValueError, match="structure_colors"):
        plot_alignment_arcs(["...."], ["AAAA"], structure_colors=[])


def test_plot_alignment_arcs_structure_colors_paint_arcs() -> None:
    """Documented contract: ``structure_colors`` customizes per-structure arc color."""
    fig = plot_alignment_arcs(["(...)", "((.))"], ["AAAAA", "AAAAA"], structure_colors=["#ff0000", "#00ff00"])
    rendered = {to_hex(color) for collection in _line_collections(fig.axes[0]) for color in collection.get_colors()}
    assert "#ff0000" in rendered
    assert "#00ff00" in rendered
    plt.close(fig)


def test_plot_alignment_arcs_conservation_paints_conserved_columns_darker() -> None:
    """``show_conservation`` should color conserved columns darker than variable ones."""
    from matplotlib.patches import Rectangle

    # Column 0 fully conserved (all 'A'); column 1 fully variable (A/C/G).
    fig = plot_alignment_arcs(["...", "...", "..."], ["AAU", "ACU", "AGU"], show_conservation=True)
    column_brightness: dict[int, list[float]] = {0: [], 1: []}
    for patch in fig.axes[0].patches:
        if not isinstance(patch, Rectangle):
            continue
        column = round(patch.get_x() + 0.45)
        if column in column_brightness:
            r, g, b, _ = patch.get_facecolor()
            column_brightness[column].append((r + g + b) / 3.0)
    assert column_brightness[0] and column_brightness[1]
    assert min(column_brightness[0]) < max(column_brightness[1])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reference / comparison rendering.
# ---------------------------------------------------------------------------


def test_plot_planar_graph_with_reference_adds_legend(nested_structure, nested_reference) -> None:
    seq, dbn = nested_structure
    fig = plot_planar_graph(dbn, seq, reference_dot_bracket=nested_reference)
    assert any(ax.get_legend() is not None for ax in fig.axes)
    plt.close(fig)


def test_plot_planar_graph_comparison_uses_documented_pair_colors() -> None:
    """TP/FP/FN pair lines must render in ``PAIR_COMPARISON_COLORS`` — a documented contract."""
    # predicted=(((...))), reference=((.....)) → TP={(0,8),(1,7)}, FP={(2,6)}, FN={}.
    fig = plot_planar_graph("(((...)))", "GGGAAAUCC", reference_dot_bracket="((.....))")
    rendered = {to_hex(color) for collection in _line_collections(fig.axes[0]) for color in collection.get_colors()}
    assert to_hex(PAIR_COMPARISON_COLORS["true_positive"]) in rendered
    assert to_hex(PAIR_COMPARISON_COLORS["false_positive"]) in rendered
    plt.close(fig)

    # FN-only: predicted misses a pair that reference has.
    fig = plot_planar_graph("((......))", "GGGAAAAUCC", reference_dot_bracket="(((....)))")
    rendered = {to_hex(color) for collection in _line_collections(fig.axes[0]) for color in collection.get_colors()}
    assert to_hex(PAIR_COMPARISON_COLORS["false_negative"]) in rendered
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_comparison_legend_coexists_with_category_track(plot_fn, nested_structure, nested_reference) -> None:
    """A category-track legend and the TP/FP/FN comparison legend must both render."""
    from matplotlib.legend import Legend

    from multimolecule.visualization.rna import BaseCategoryTrack

    seq, dbn = nested_structure
    track = BaseCategoryTrack(["stem"] * len(dbn))
    fig = plot_fn(dbn, seq, reference_dot_bracket=nested_reference, tracks=[track])
    legend_texts = [
        {text.get_text() for text in artist.get_texts()}
        for artist in fig.axes[0].get_children()
        if isinstance(artist, Legend)
    ]
    assert {"TP", "FP", "FN"} in legend_texts
    assert {"stem"} in legend_texts
    plt.close(fig)


# ---------------------------------------------------------------------------
# Arc diagram.
# ---------------------------------------------------------------------------


def test_plot_arc_diagram_unpaired_structure_keeps_usable_figure_height() -> None:
    fig = plot_arc_diagram("." * 100)
    assert fig.get_size_inches()[1] >= 1.0
    plt.close(fig)


def test_plot_arc_diagram_dual_mode_y_axis_spans_baseline(nested_structure, nested_reference) -> None:
    seq, dbn = nested_structure
    fig = plot_arc_diagram(dbn, seq, reference_dot_bracket=nested_reference, comparison_style="dual")
    y_min, y_max = fig.axes[0].get_ylim()
    assert y_min < 0 < y_max
    plt.close(fig)


def test_plot_arc_diagram_overlay_keeps_y_axis_above_baseline(nested_structure, nested_reference) -> None:
    seq, dbn = nested_structure
    fig = plot_arc_diagram(dbn, seq, reference_dot_bracket=nested_reference)
    y_min, y_max = fig.axes[0].get_ylim()
    assert y_min > -1.0
    assert y_max > 0
    plt.close(fig)


def test_plot_arc_diagram_rejects_unknown_comparison_style(nested_structure) -> None:
    seq, dbn = nested_structure
    with pytest.raises(ValueError, match="comparison_style"):
        plot_arc_diagram(dbn, seq, reference_dot_bracket=dbn, comparison_style="???")


def test_plot_arc_diagram_dual_mode_rejects_reference_length_mismatch(nested_structure) -> None:
    seq, dbn = nested_structure
    with pytest.raises(ValueError, match="reference_dot_bracket length"):
        plot_arc_diagram(dbn, seq, reference_dot_bracket=dbn + ".", comparison_style="dual")


def test_plot_arc_diagram_accepts_taller_height_scale(nested_structure) -> None:
    seq, dbn = nested_structure
    fig = plot_arc_diagram(dbn, seq, arc_height_scale=1.5)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_arc_diagram_rejects_zero_height_scale(nested_structure) -> None:
    seq, dbn = nested_structure
    with pytest.raises(ValueError, match="arc_height_scale"):
        plot_arc_diagram(dbn, seq, arc_height_scale=0.0)


def test_plot_arc_diagram_palette_colors_base_labels() -> None:
    """Arc-diagram base labels are colored by nucleotide — matches planar / circular defaults."""
    fig = plot_arc_diagram("(..)", "ACGU")
    label_colors = {to_hex(t.get_color()) for t in fig.axes[0].texts if t.get_text() in set("ACGU")}
    assert "#000000" not in label_colors
    assert len(label_colors) >= 2
    plt.close(fig)


def test_plot_arc_diagram_base_color_overrides_apply_with_default_palette() -> None:
    fig = plot_arc_diagram("....", "AAAA", base_colors={"A": "#ff0000"})
    label_colors = {to_hex(t.get_color()) for t in fig.axes[0].texts if t.get_text() == "A"}
    assert label_colors == {"#ff0000"}
    plt.close(fig)


# ---------------------------------------------------------------------------
# Circular diagram.
# ---------------------------------------------------------------------------


def test_plot_circular_diagram_shows_5_prime_3_prime_by_default(nested_structure) -> None:
    seq, dbn = nested_structure
    fig = plot_circular_diagram(dbn, seq)
    labels = {t.get_text(): t for t in fig.axes[0].texts}
    assert "5'" in labels
    assert "3'" in labels
    plt.close(fig)


def test_plot_circular_diagram_hides_end_labels_when_show_ends_false(nested_structure) -> None:
    seq, dbn = nested_structure
    fig = plot_circular_diagram(dbn, seq, show_ends=False)
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert "5'" not in texts
    assert "3'" not in texts
    plt.close(fig)


def test_plot_circular_diagram_mfe_pair_color_override(nested_structure) -> None:
    from multimolecule.visualization.rna import PairProbabilityTrack

    seq, dbn = nested_structure
    matrix = np.zeros((len(dbn), len(dbn)))
    fig = plot_circular_diagram(
        dbn,
        seq,
        tracks=[PairProbabilityTrack(matrix=matrix, threshold=0.5)],
        mfe_pair_color="#123456",
    )
    rendered = {to_hex(color) for collection in _line_collections(fig.axes[0]) for color in collection.get_colors()}
    assert "#123456" in rendered
    plt.close(fig)


def test_plot_circular_diagram_chord_curvature_accepts_numeric_and_auto(nested_structure) -> None:
    """``chord_curvature`` accepts ``"auto"`` and any value in ``[0.0, 1.0]``."""
    seq, dbn = nested_structure
    for value in (0.0, 0.5, 1.0, "auto"):
        fig = plot_circular_diagram(dbn, seq, chord_curvature=value)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Pseudoknot tier styling — user-visible coloring of crossing pairs.
# ---------------------------------------------------------------------------


def test_arc_and_circular_pseudoknot_tiers_use_primary_bracket() -> None:
    """When dot-bracket has explicit ``[`` brackets, the renderer treats ``()`` as primary."""
    arc_fig = plot_arc_diagram("[(])", pair_color="#111111")
    arc_lines = _line_collection_segment_colors(arc_fig.axes[0])
    assert arc_lines[(1, 3)] == "#111111"
    assert arc_lines[(0, 2)] == PSEUDOKNOT_TIER_COLORS[0]
    plt.close(arc_fig)

    circular_fig = plot_circular_diagram("[(])", pair_color="#111111")
    chord_colors = [
        to_hex(color)
        for collection in _line_collections(circular_fig.axes[0])
        for color in collection.get_colors()
        if collection.get_zorder() == 2
    ]
    assert chord_colors.count("#111111") == 1
    assert chord_colors.count(PSEUDOKNOT_TIER_COLORS[0]) == 1
    plt.close(circular_fig)


def test_circular_pseudoknot_tier_toggle_uses_plain_pair_color() -> None:
    """``show_pseudoknot_tiers=False`` paints every pair with ``pair_color``."""
    from multimolecule.utils.rna.secondary_structure import dot_bracket_to_pairs

    dot_bracket = "((..[[..))..]]"
    fig = plot_circular_diagram(dot_bracket, pair_color="#111111", show_pseudoknot_tiers=False)
    line_colors = [
        to_hex(color)
        for collection in _line_collections(fig.axes[0])
        for color in collection.get_colors()
        if collection.get_zorder() == 2
    ]
    assert line_colors.count("#111111") == len(dot_bracket_to_pairs(dot_bracket))
    assert PSEUDOKNOT_TIER_COLORS[0] not in line_colors
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_pseudoknot_tier_styling_toggle(plot_fn, pseudoknot_structure) -> None:
    seq, dbn = pseudoknot_structure
    for show in (True, False):
        fig = plot_fn(dbn, seq, show_pseudoknot_tiers=show)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Contact map.
# ---------------------------------------------------------------------------


def test_plot_contact_map_dual_triangle_from_dot_bracket(random_contact_map) -> None:
    length = random_contact_map.shape[0]
    fig = plot_contact_map(random_contact_map, reference="." * length)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_contact_map_rejects_shape_mismatch(random_contact_map) -> None:
    with pytest.raises(ValueError, match="reference shape"):
        plot_contact_map(random_contact_map, reference=np.zeros((5, 5)))


def test_plot_contact_map_dual_triangle_assignment_is_correct() -> None:
    """Documented contract: upper triangle is ``contact_map``, lower triangle is ``reference``."""
    length = 6
    predicted = np.ones((length, length))
    reference = np.zeros((length, length))
    fig = plot_contact_map(predicted, reference=reference)
    displayed = fig.axes[0].images[0].get_array()
    np.testing.assert_array_equal(displayed[np.triu_indices(length, k=1)], 1.0)
    np.testing.assert_array_equal(displayed[np.tril_indices(length, k=-1)], 0.0)
    plt.close(fig)


def test_contact_map_track_sidebars_span_full_data_width() -> None:
    """Regression: the top sidebar must span the full axes width so column i in the bar
    sits over column i in the matrix. A colorbar overlapping the rightmost cell is acceptable;
    column misalignment is not.
    """
    from multimolecule.visualization.rna import BaseValueTrack

    fig = plot_contact_map(np.eye(12), colorbar=True, tracks=[BaseValueTrack(np.linspace(0.0, 1.0, 12))])
    sidebar_axes = [child for child in fig.axes[0].child_axes if child.get_gid() != "multimolecule-attached-colorbar"]
    top_sidebar = max(sidebar_axes, key=lambda child: child.get_position().y0)
    pos = top_sidebar.get_position()
    assert pos.x0 == pytest.approx(0.0, abs=1e-9)
    assert pos.x1 == pytest.approx(1.0, abs=1e-9)
    plt.close(fig)
