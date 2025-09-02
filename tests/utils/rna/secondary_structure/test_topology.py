# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
#
# This file is part of MultiMolecule.
#
# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


import numpy as np
import pytest
import torch

from multimolecule.utils.rna.secondary_structure import topology


def test_structure_single_nucleotide_edges() -> None:
    structure = topology.RnaSecondaryStructure("A", ".")

    assert structure.edge_index.shape == (0, 2)
    assert structure.edge_features["type"].shape == (0, 1)
    assert structure.pairs.numel() == 0
    assert structure.stems == []


def test_structure_no_pairs_loops_and_annotations() -> None:
    structure = topology.RnaSecondaryStructure("ACGU", "....")

    assert structure.pairs.shape == (0, 2)
    assert structure.primary_pairs.shape == (0, 2)
    assert structure.pseudoknot_pairs.shape == (0, 2)

    labels = structure.loop_labels
    assert labels.tolist() == [int(topology.LoopType.END)] * 4

    loops = structure.loops
    assert loops.ends[0].tolist() == [0, 1, 2, 3]
    assert loops.hairpins == []
    assert loops.bulges == []
    assert loops.internals == []
    assert loops.branches == []
    assert loops.externals == []

    assert structure.structural_annotation == "EEEE"
    assert structure.functional_annotation == "NNNN"
    assert structure.stems_ptr.tolist() == [0]


def test_structure_pseudoknot_edges_and_crossings() -> None:
    structure = topology.RnaSecondaryStructure("ACGU", "([)]")

    assert structure.primary_pairs.tolist() == [[0, 2]]
    assert structure.pseudoknot_pairs.tolist() == [[1, 3]]
    assert structure.crossing_pairs.tolist() == [[0, 2], [1, 3]]

    edge_types = structure.edge_features["type"].reshape(-1)
    assert int((edge_types == topology.EdgeType.BACKBONE.value).sum().item()) == 3
    assert int((edge_types == topology.EdgeType.PRIMARY_PAIRS.value).sum().item()) == 1
    assert int((edge_types == topology.EdgeType.PSEUDOKNOT_PAIR.value).sum().item()) == 1

    assert structure.primary_stems_ptr.tolist() == [0, 1]
    assert structure.pseudoknot_stems_ptr.tolist() == [0, 1]
    assert structure.stems_ptr.tolist() == [0, 1, 2]

    assert structure.functional_annotation == "NKNK"


def test_find_stem_ptr_contiguous() -> None:
    pairs = torch.tensor([[0, 5], [1, 4], [2, 3]], dtype=torch.long)
    stem_ptr = topology._find_stem_ptr(pairs)
    assert stem_ptr.tolist() == [0, 3]

    stems = topology._stems_from_ptr(pairs, stem_ptr)
    assert len(stems) == 1
    assert stems[0].tolist() == [[0, 5], [1, 4], [2, 3]]


def test_read_dot_brackets_paths_and_errors() -> None:
    pairs, primary_pairs, pk_pairs, has_pk, primary_open = topology._read_dot_brackets("((..))")
    assert not has_pk
    assert pairs.tolist() == [[0, 5], [1, 4]]
    assert primary_pairs.tolist() == [[0, 5], [1, 4]]
    assert pk_pairs.shape == (0, 2)
    assert primary_open.tolist() == [5, 4, -1, -1, -1, -1]

    pairs, primary_pairs, pk_pairs, has_pk, primary_open = topology._read_dot_brackets("([)]")
    assert has_pk
    assert pairs.tolist() == [[0, 2], [1, 3]]
    assert primary_pairs.tolist() == [[0, 2]]
    assert pk_pairs.tolist() == [[1, 3]]
    assert primary_open.tolist() == [2, -1, -1, -1]

    with pytest.raises(ValueError, match="Unmatched symbol"):
        topology._read_dot_brackets(")")
    with pytest.raises(ValueError, match="Invalid symbol"):
        topology._read_dot_brackets("1")
    with pytest.raises(ValueError, match="Unmatched symbol"):
        topology._read_dot_brackets("(")


@pytest.mark.parametrize(
    "open_to_close, expected_codes, expected_nonempty",
    [
        (
            np.full(6, -1, dtype=np.int64),
            {topology.LoopType.END},
            {topology.LoopType.END},
        ),
        (
            np.array([-1, 4, -1, -1, -1, -1], dtype=np.int64),
            {topology.LoopType.END, topology.LoopType.HAIRPIN},
            {topology.LoopType.HAIRPIN},
        ),
        (
            np.array([6, -1, 5, -1, -1, -1, -1], dtype=np.int64),
            {topology.LoopType.BULGE, topology.LoopType.HAIRPIN},
            {topology.LoopType.BULGE},
        ),
        (
            np.array([7, -1, 5, -1, -1, -1, -1, -1], dtype=np.int64),
            {topology.LoopType.INTERNAL, topology.LoopType.HAIRPIN},
            {topology.LoopType.INTERNAL},
        ),
        (
            np.array([9, -1, 4, -1, -1, -1, 8, -1, -1, -1], dtype=np.int64),
            {topology.LoopType.BRANCH, topology.LoopType.HAIRPIN},
            {topology.LoopType.BRANCH},
        ),
        (
            np.array([-1, 4, -1, -1, -1, -1, 9, -1, -1, -1, -1], dtype=np.int64),
            {topology.LoopType.END, topology.LoopType.EXTERNAL, topology.LoopType.HAIRPIN},
            {topology.LoopType.EXTERNAL},
        ),
    ],
)
def test_build_loop_labels_and_segments(open_to_close, expected_codes, expected_nonempty) -> None:
    labels, segments = topology._build_loop_labels_and_segments(open_to_close, len(open_to_close), torch.device("cpu"))
    codes = {code for _, _, code in segments}
    assert expected_codes.issubset(codes)

    loops = topology._segments_to_loops(segments, len(open_to_close), labels.device)
    loop_map = {
        topology.LoopType.HAIRPIN: loops.hairpins,
        topology.LoopType.BULGE: loops.bulges,
        topology.LoopType.INTERNAL: loops.internals,
        topology.LoopType.BRANCH: loops.branches,
        topology.LoopType.EXTERNAL: loops.externals,
        topology.LoopType.END: loops.ends,
    }
    for loop_type in expected_nonempty:
        assert loop_map[loop_type]


def test_build_loop_labels_empty_length() -> None:
    labels, segments = topology._build_loop_labels_and_segments(np.empty((0,), dtype=np.int64), 0, torch.device("cpu"))
    assert labels.numel() == 0
    assert segments == []

    loops = topology._segments_to_loops([], 0, labels.device)
    assert loops.loops == []
