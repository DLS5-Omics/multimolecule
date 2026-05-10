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
from matplotlib.figure import Figure

from multimolecule.utils.rna.secondary_structure import dot_bracket_to_contact_map
from multimolecule.visualization.rna import (
    STRUCTURAL_CLASS_PALETTE,
    BaseCategoryTrack,
    BaseValueTrack,
    LeontisWesthof,
    PairAnnotationTrack,
    PairProbabilityTrack,
    RegionTrack,
    SequenceDiffTrack,
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


# ---------------------------------------------------------------------------
# Each track composes with every structure plot.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_base_value_track_composes_with_structure_plots(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    fig = plot_fn(dbn, seq, tracks=[BaseValueTrack(values=np.linspace(0.0, 1.0, len(dbn)))])
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_base_value_track_length_mismatch_raises(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    with pytest.raises(ValueError, match="length"):
        plot_fn(dbn, seq, tracks=[BaseValueTrack(values=[0.0, 1.0])])


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_region_track_composes_with_structure_plots(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    track = RegionTrack(regions=[(0, 8, "SL1"), (12, 19, "SL2")], colors=["#ff7f0e", "#1f77b4"])
    fig = plot_fn(dbn, seq, tracks=[track])
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_base_category_track_composes_with_structure_plots(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    categories = ["A" if i % 2 else "B" for i in range(len(dbn))]
    fig = plot_fn(dbn, seq, tracks=[BaseCategoryTrack(categories=categories)])
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_pair_probability_track_composes_with_structure_plots(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    length = len(dbn)
    matrix = np.asarray(dot_bracket_to_contact_map(dbn), dtype=float)
    matrix = np.clip((matrix + matrix.T) / 2, 0.0, 1.0)
    fig = plot_fn(dbn, seq, tracks=[PairProbabilityTrack(matrix=matrix, threshold=0.1)])
    assert isinstance(fig, Figure)
    assert length == matrix.shape[0]
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_sequence_diff_track_composes_with_structure_plots(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    track = SequenceDiffTrack(reference_sequence="A" + seq[1:])
    fig = plot_fn(dbn, seq, tracks=[track])
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("plot_fn", STRUCTURE_PLOTS)
def test_tracks_stack_in_a_single_plot_call(plot_fn, nested_structure) -> None:
    seq, dbn = nested_structure
    length = len(dbn)
    tracks = [
        BaseValueTrack(values=np.linspace(0, 1, length)),
        RegionTrack(regions=[(0, 8, "SL1")]),
        SequenceDiffTrack(reference_sequence="A" + seq[1:]),
    ]
    fig = plot_fn(dbn, seq, tracks=tracks)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tracks compose with contact_map.
# ---------------------------------------------------------------------------


def test_base_value_track_renders_on_contact_map(random_contact_map) -> None:
    track = BaseValueTrack(values=np.linspace(0.0, 1.0, random_contact_map.shape[0]))
    fig = plot_contact_map(random_contact_map, colorbar=False, tracks=[track])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_base_category_track_renders_on_contact_map(random_contact_map) -> None:
    length = random_contact_map.shape[0]
    track = BaseCategoryTrack(categories=["A" if i % 2 else "B" for i in range(length)])
    fig = plot_contact_map(random_contact_map, colorbar=False, tracks=[track])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_region_track_renders_on_contact_map(random_contact_map) -> None:
    fig = plot_contact_map(random_contact_map, colorbar=False, tracks=[RegionTrack(regions=[(1, 3, "region")])])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_pair_probability_track_is_no_op_on_contact_map(random_contact_map) -> None:
    """The matrix already encodes pair probabilities, so the track adds no overlay."""
    track = PairProbabilityTrack(matrix=np.zeros_like(random_contact_map))
    fig = plot_contact_map(random_contact_map, colorbar=False, tracks=[track])
    assert len(fig.axes[0].patches) == 0
    plt.close(fig)


def test_sequence_diff_track_highlights_changed_ticks_on_contact_map(random_contact_map) -> None:
    """At positions where the plotted sequence differs from the reference, both axis tick labels
    use ``highlight_color`` and render bold — the user-visible riboSNitch signal.
    """
    length = random_contact_map.shape[0]
    sequence = "C" + "A" * (length - 1)
    track = SequenceDiffTrack(reference_sequence="A" * length, highlight_color="#123456")
    fig = plot_contact_map(random_contact_map, sequence=sequence, tracks=[track])
    ax = fig.axes[0]
    x_labels = {label.get_text(): label for label in ax.get_xticklabels()}
    y_labels = {label.get_text(): label for label in ax.get_yticklabels()}
    assert x_labels["0"].get_color() == "#123456"
    assert y_labels["0"].get_color() == "#123456"
    assert x_labels["0"].get_fontweight() == "bold"
    plt.close(fig)


def test_sequence_diff_track_requires_sequence_on_contact_map(random_contact_map) -> None:
    track = SequenceDiffTrack(reference_sequence="A" * random_contact_map.shape[0])
    with pytest.raises(ValueError, match="sequence is required"):
        plot_contact_map(random_contact_map, tracks=[track])


# ---------------------------------------------------------------------------
# Documented track validation contracts.
# ---------------------------------------------------------------------------


def test_pair_probability_track_rejects_shape_mismatch(nested_structure) -> None:
    seq, dbn = nested_structure
    with pytest.raises(ValueError, match="matrix shape"):
        plot_planar_graph(dbn, seq, tracks=[PairProbabilityTrack(matrix=np.zeros((5, 5)))])


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, np.nan, np.inf])
def test_pair_probability_track_rejects_invalid_probabilities(bad_value) -> None:
    matrix = np.zeros((2, 2), dtype=float)
    matrix[0, 1] = bad_value
    with pytest.raises(ValueError, match="matrix values"):
        PairProbabilityTrack(matrix=matrix).validate(2)


def test_pair_probability_track_rejects_invalid_haze_interval() -> None:
    track = PairProbabilityTrack(matrix=np.zeros((5, 5)), threshold=0.2, haze_threshold=0.2)
    with pytest.raises(ValueError, match="haze_threshold"):
        track.validate(5)


def test_pair_probability_track_haze_renders(nested_structure) -> None:
    seq, dbn = nested_structure
    length = len(dbn)
    rng = np.random.default_rng(0)
    matrix = rng.random((length, length)) * 0.3
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    track = PairProbabilityTrack(matrix=matrix, threshold=0.2, haze_threshold=0.02)
    for plot_fn in (plot_planar_graph, plot_arc_diagram, plot_circular_diagram):
        fig = plot_fn(dbn, seq, tracks=[track])
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Constructors that build tracks from RNA inputs.
# ---------------------------------------------------------------------------


def test_base_category_track_from_topology_returns_known_loop_kinds(nested_structure) -> None:
    _, dbn = nested_structure
    track = BaseCategoryTrack.from_topology(dbn)
    assert {"hairpin", "stem", "external"}.issubset(set(track.categories))


def test_base_category_track_from_topology_structural_class_mode(nested_structure) -> None:
    _, dbn = nested_structure
    track = BaseCategoryTrack.from_topology(dbn, mode="structural_class")
    assert set(track.categories).issubset(set("SHBIMXE"))
    assert track.palette is not None
    assert track.palette["H"] == STRUCTURAL_CLASS_PALETTE["H"]


def test_base_category_track_from_topology_rejects_unknown_mode(nested_structure) -> None:
    _, dbn = nested_structure
    with pytest.raises(ValueError, match="Unknown mode"):
        BaseCategoryTrack.from_topology(dbn, mode="not-a-mode")


def test_base_category_track_from_sequence(nested_structure) -> None:
    seq, _ = nested_structure
    track = BaseCategoryTrack.from_sequence(seq)
    assert track.categories == [c.upper() for c in seq]
    assert track.palette is not None
    assert {"A", "C", "G", "U"}.issubset(track.palette)


def test_base_category_track_from_sequence_maps_unknown_bases_to_n() -> None:
    """Unknown / non-canonical bases collapse to ``N`` so renderers always have a palette entry."""
    track = BaseCategoryTrack.from_sequence("AXN")
    assert track.categories == ["A", "N", "N"]


def test_base_category_track_from_sequence_rejects_unknown_palette() -> None:
    with pytest.raises(ValueError, match="Unknown nucleotide palette"):
        BaseCategoryTrack.from_sequence("AAAA", palette="not-real")


def test_region_track_from_topology_finds_hairpins(nested_structure) -> None:
    _, dbn = nested_structure  # fixture has two hairpins
    track = RegionTrack.from_topology(dbn, loop_kinds=("hairpin",))
    assert len(track.regions) == 2
    for start, stop, label in track.regions:
        assert label == "Hairpin"
        assert start <= stop


def test_region_track_from_topology_rejects_unknown_loop_kind(nested_structure) -> None:
    _, dbn = nested_structure
    with pytest.raises(ValueError, match="Unknown loop_kinds"):
        RegionTrack.from_topology(dbn, loop_kinds=("hairpins",))


# ---------------------------------------------------------------------------
# Leontis-Westhof annotations.
# ---------------------------------------------------------------------------


def test_leontis_westhof_validates_edges_and_orientation() -> None:
    with pytest.raises(ValueError, match="edge_5p"):
        LeontisWesthof(edge_5p="X", edge_3p="WC")
    with pytest.raises(ValueError, match="edge_3p"):
        LeontisWesthof(edge_5p="WC", edge_3p="Q")
    with pytest.raises(ValueError, match="orientation"):
        LeontisWesthof(edge_5p="WC", edge_3p="WC", orientation="other")


def test_pair_annotation_track_rejects_reversed_key() -> None:
    with pytest.raises(ValueError, match="must be ordered"):
        PairAnnotationTrack(annotations={(5, 1): LeontisWesthof("WC", "WC")})


def test_pair_annotation_track_rejects_self_pair() -> None:
    with pytest.raises(ValueError, match="self-pair"):
        PairAnnotationTrack(annotations={(3, 3): LeontisWesthof("WC", "WC")})


def test_pair_annotation_track_rejects_non_lw_value() -> None:
    with pytest.raises(TypeError, match="LeontisWesthof"):
        PairAnnotationTrack(annotations={(1, 5): "WC-WC"})  # type: ignore[dict-item]


def test_pair_annotation_track_rejects_pair_outside_structure(nested_structure) -> None:
    _, dbn = nested_structure
    track = PairAnnotationTrack(annotations={(0, len(dbn)): LeontisWesthof("WC", "WC")})
    with pytest.raises(ValueError, match="outside structure length"):
        plot_planar_graph(dbn, None, tracks=[track])


def test_pair_annotation_track_rejects_pair_absent_from_structure(nested_structure) -> None:
    _, dbn = nested_structure
    track = PairAnnotationTrack(annotations={(0, 7): LeontisWesthof("WC", "WC")})
    with pytest.raises(ValueError, match="not present"):
        plot_planar_graph(dbn, None, tracks=[track])


def test_pair_annotation_track_renders_on_planar_and_circular(nested_structure) -> None:
    seq, dbn = nested_structure
    track = PairAnnotationTrack(
        annotations={
            (0, 8): LeontisWesthof("WC", "WC", "cis"),
            (1, 7): LeontisWesthof("H", "S", "trans"),
        },
    )
    for plot_fn in (plot_planar_graph, plot_circular_diagram):
        fig = plot_fn(dbn, seq, tracks=[track])
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# NaN handling — documented contract for BaseValueTrack.
# ---------------------------------------------------------------------------


def test_base_value_track_renders_nan_as_neutral_gray(nested_structure) -> None:
    """The docstring promises NaN values render as a neutral gray ``(0.65, 0.65, 0.65, 1.0)``."""
    seq, dbn = nested_structure
    length = len(dbn)
    values = np.linspace(0.0, 1.0, length).tolist()
    nan_index = 5
    values[nan_index] = float("nan")
    fig = plot_planar_graph(dbn, seq, tracks=[BaseValueTrack(values=values, cmap="viridis")])
    bead_scatter = next(coll for coll in fig.axes[0].collections if np.asarray(coll.get_offsets()).shape == (length, 2))
    facecolors = np.asarray(bead_scatter.get_facecolors())
    np.testing.assert_allclose(facecolors[nan_index], (0.65, 0.65, 0.65, 1.0), atol=1e-6)
    plt.close(fig)


def test_base_value_track_renders_all_nan_without_crash(nested_structure) -> None:
    seq, dbn = nested_structure
    fig = plot_planar_graph(dbn, seq, tracks=[BaseValueTrack(values=[float("nan")] * len(dbn))])
    assert isinstance(fig, Figure)
    plt.close(fig)
