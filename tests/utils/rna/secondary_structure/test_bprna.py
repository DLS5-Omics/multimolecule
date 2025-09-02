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

from multimolecule.utils.rna.secondary_structure import StemEdgeType, bprna, notations
from multimolecule.utils.rna.secondary_structure import pairs as pairs_utils
from multimolecule.utils.rna.secondary_structure import topology
from multimolecule.utils.rna.secondary_structure.bprna import STRUCTURE_TYPE_KEYS
from tests.utils.rna.secondary_structure.bprna_cases import REFERENCE_CASES
from tests.utils.rna.secondary_structure.conftest import as_set


def test_bprna_empty_structure() -> None:
    structure = topology.RnaSecondaryStructureTopology("", "")
    assert bprna.annotate_structure(structure) == ""
    assert bprna.annotate_function(structure) == ""
    assert bprna.annotate(structure) == ("", "")
    types = bprna.structure_types(structure)
    assert set(types.keys()) == set(STRUCTURE_TYPE_KEYS)
    assert all(value == [] for value in types.values())


def test_bprna_no_pairs_annotation() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    structural_annotation, functional_annotation = bprna.annotate(structure)
    assert structural_annotation == "EEEE"
    assert functional_annotation == "NNNN"


def test_bprna_hairpin_annotation() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACG", "(.)")
    assert bprna.annotate_structure(structure) == "SHS"
    assert bprna.annotate_function(structure) == "NNN"


def test_bprna_pseudoknot_annotation() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    structural_annotation, functional_annotation = bprna.annotate(structure)
    assert structural_annotation == "ESHS"
    assert functional_annotation == "KNKN"


def test_bprna_pseudoknot_function_annotation() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    assert bprna.annotate_function(structure) == "KNKN"


def test_bprna_structure_types_external_with_lowercase_sequence() -> None:
    sequence = "acguacguacguac"
    dot_bracket = "((..))..((..))"
    structure = topology.RnaSecondaryStructureTopology(sequence, dot_bracket)
    types = bprna.structure_types(structure)
    assert types["X"]
    assert any(line.startswith("X") for line in types["X"])


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=[case["id"] for case in REFERENCE_CASES])
def test_bprna_reference_cases(case: dict) -> None:
    structure = topology.RnaSecondaryStructureTopology(case["sequence"], case["dot_bracket"])
    structural_annotation, functional_annotation = bprna.annotate(structure)
    assert structural_annotation == case["structural"], case["id"]
    assert functional_annotation == case["functional"], case["id"]


def test_bprna_structure_types_hairpin() -> None:
    structure = topology.RnaSecondaryStructureTopology("GCU", "(.)")
    types = bprna.structure_types(structure)

    assert set(types.keys()) == set(STRUCTURE_TYPE_KEYS)
    assert len(types["S"]) == 1
    assert len(types["H"]) == 1
    assert len(types["NCBP"]) == 0
    assert len(types["SEGMENTS"]) == 1
    assert "1..1" in types["S"][0]
    assert "3..3" in types["S"][0]
    assert "2..2" in types["H"][0]
    assert types["H"][0].endswith("\n")
    assert types["S"][0].endswith("\n")
    assert types["SEGMENTS"][0].startswith("segment1 1bp 1..1")


def test_bprna_helpers_and_structure_types() -> None:
    with pytest.raises(ValueError):
        bprna.BpRnaSecondaryStructureTopology("AC", dot_bracket="...")
    with pytest.raises(ValueError):
        bprna.BpRnaSecondaryStructureTopology("AC")

    structure = topology.RnaSecondaryStructureTopology("AC", "..")
    with pytest.raises(ValueError):
        bprna.BpRnaSecondaryStructureTopology("AG", topology=structure)

    assert bprna.BpRnaSecondaryStructureTopology._safe_base("AC", -1) == ""
    assert bprna.BpRnaSecondaryStructureTopology._safe_base("AC", 2) == ""
    assert bprna.BpRnaSecondaryStructureTopology._slice_sequence("AC", -1, 1) == ""
    assert bprna.BpRnaSecondaryStructureTopology._slice_sequence("AC", 1, 0) == ""

    empty = bprna.BpRnaSecondaryStructureTopology("", dot_bracket="")
    assert empty.structural_annotation == ""
    assert empty.functional_annotation == ""

    bulge = bprna.BpRnaSecondaryStructureTopology("AAAAA", dot_bracket="(.())")
    internal = bprna.BpRnaSecondaryStructureTopology("A" * 10, dot_bracket="((..(.).))")
    multi_pairs = [(0, 11), (2, 3), (5, 6), (8, 9)]
    multi_dbn = notations.pairs_to_dot_bracket(multi_pairs, length=12)
    multi = bprna.BpRnaSecondaryStructureTopology("A" * 12, dot_bracket=multi_dbn)

    assert bulge.structure_types["B"]
    assert internal.structure_types["I"]
    assert multi.structure_types["M"]

    assert bprna.annotate_structure(structure)
    assert bprna.annotate_function(structure)
    assert bprna.structure_types(structure)
    assert bprna.annotate(structure)


def test_bprna_ncbp_numbering_across_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    types = bprna.structure_types(structure)
    ncbp = types["NCBP"]
    assert len(ncbp) == 2
    numbers = [int(line.split()[0][4:]) for line in ncbp]
    assert numbers == [1, 2]


def test_pairmap_init_shape_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        bprna.PairMap(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="shape"):
        bprna.PairMap([(0, 1, 2)])


def test_pairmap_rejects_multiple_partners() -> None:
    with pytest.raises(ValueError, match="multiple partners"):
        bprna.PairMap([(0, 2), (0, 3)])
    with pytest.raises(ValueError, match="multiple partners"):
        bprna.PairMap([(0, 2), (3, 2)])


def test_pairmap_accessors() -> None:
    pair_map = bprna.PairMap([(0, 3), (1, 2)])
    assert 0 in pair_map
    assert pair_map[0] == 3
    assert set(pair_map.keys()) == {0, 1, 2, 3}
    assert dict(pair_map.items())[1] == 2
    assert sorted(pair_map) == [0, 1, 2, 3]


def test_pairmap_loop_and_stem_checks() -> None:
    pair_map = bprna.PairMap([(0, 3), (1, 2)])
    link_map = bprna.PairMap([(3, 4)])
    assert not pair_map.is_loop_linked(0, 0)
    assert link_map.is_loop_linked(2, 5)
    assert pair_map.to_list() == [3, 2, 1, 0]


def test_pairmap_empty_cache_arrays() -> None:
    empty_map = bprna.PairMap([])
    assert empty_map.pairs == []
    assert empty_map.segments == []


def test_segment_topology_graph_helpers() -> None:
    empty_topology = bprna.BpRnaSecondaryStructureTopology("AC", np.empty((0, 2), dtype=int))
    assert empty_topology.topology.nested_stem_segments.edges == []

    pairs_np = np.array([[0, 5], [1, 4], [6, 9], [7, 8]])
    segment_topology = bprna.BpRnaSecondaryStructureTopology("ACGUACGUAC", pairs_np)
    assert len(segment_topology) == 10
    assert segment_topology.topology.nested_stem_segments.edges
    assert all(isinstance(edge.type, StemEdgeType) for edge in segment_topology.topology.nested_stem_segments.edges)


def test_segments_helpers_consistency() -> None:
    pairs_np = np.array([[0, 6], [1, 5], [2, 8], [3, 7]])
    pair_map = bprna.PairMap(pairs_np)
    start_i, start_j, lengths = pairs_utils.pairs_to_helix_segment_arrays(pairs_np)
    segments = []
    count = int(start_i.numel()) if hasattr(start_i, "numel") else int(start_i.size)
    for idx in range(count):
        seg_len = int(lengths[idx])
        if seg_len <= 0:
            continue
        seg_start_i = int(start_i[idx])
        seg_start_j = int(start_j[idx])
        segments.append([(seg_start_i + offset, seg_start_j - offset) for offset in range(seg_len)])
    assert segments == pair_map.segments
    assert len(segments) == 2
    rebuilt = pairs_utils.segment_arrays_to_pairs(segments, empty=pairs_np[:0])
    assert as_set(rebuilt) == as_set(pairs_np)
