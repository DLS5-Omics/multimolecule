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

from __future__ import annotations

from typing import List, Set

import pytest
import torch

from multimolecule.utils.rna.secondary_structure import Pair, pairs_to_dot_bracket, topology
from tests.utils.rna.secondary_structure.conftest import as_list, as_tuple_list


def _nonempty_loop_types(loops: topology.LoopSegments) -> Set[topology.LoopSegmentType]:
    types: Set[topology.LoopSegmentType] = set()
    segments = loops.segments
    if any(segment.kind == topology.LoopSegmentType.HAIRPIN for segment in segments):
        types.add(topology.LoopSegmentType.HAIRPIN)
    if any(segment.kind == topology.LoopSegmentType.BULGE for segment in segments):
        types.add(topology.LoopSegmentType.BULGE)
    if any(segment.kind == topology.LoopSegmentType.INTERNAL for segment in segments):
        types.add(topology.LoopSegmentType.INTERNAL)
    if any(segment.kind == topology.LoopSegmentType.BRANCH for segment in segments):
        types.add(topology.LoopSegmentType.BRANCH)
    if any(segment.kind == topology.LoopSegmentType.EXTERNAL for segment in segments):
        types.add(topology.LoopSegmentType.EXTERNAL)
    if any(segment.kind == topology.LoopSegmentType.END for segment in segments):
        types.add(topology.LoopSegmentType.END)
    return types


def _loop_spans(loop: topology.Loop) -> List[Pair]:
    return [(span.start, span.stop) for span in loop.spans]


def _loop_signature(loops: topology.Loops) -> List[tuple]:
    return [(loop.kind, _loop_spans(loop)) for loop in loops]


def test_compute_loop_spans_list_input() -> None:
    open_to_close = [5, 4, -1, -1, -1, -1]
    roots, children, intervals = topology.compute_loop_spans(open_to_close, len(open_to_close))
    assert roots == [0]
    assert children[0] == [1]
    assert any(kind == topology.LoopSegmentType.HAIRPIN for _, _, kind, _ in intervals)


def test_structure_single_nucleotide_edges() -> None:
    structure = topology.RnaSecondaryStructureTopology("A", ".")

    assert structure.edge_index.shape == (0, 2)
    assert structure.edge_features["type"].shape == (0, 1)
    assert structure.pairs().numel() == 0
    assert structure.nested_stem_segments.segments == []


def test_structure_no_pairs_loops() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")

    assert structure.pairs().shape == (0, 2)
    assert structure.nested_pairs.shape == (0, 2)
    assert structure.pseudoknot_pairs.shape == (0, 2)

    loop_segments = structure.nested_loop_segments
    end_segments = [segment for segment in loop_segments.segments if segment.kind == topology.LoopSegmentType.END]
    assert len(end_segments) == 1
    assert (end_segments[0].start, end_segments[0].stop) == (0, 3)
    assert end_segments[0].side is None
    assert all(segment.kind != topology.LoopSegmentType.HAIRPIN for segment in loop_segments.segments)
    assert all(segment.kind != topology.LoopSegmentType.BULGE for segment in loop_segments.segments)
    assert all(segment.kind != topology.LoopSegmentType.INTERNAL for segment in loop_segments.segments)
    assert all(segment.kind != topology.LoopSegmentType.BRANCH for segment in loop_segments.segments)
    assert all(segment.kind != topology.LoopSegmentType.EXTERNAL for segment in loop_segments.segments)

    assert structure.nested_stem_segments.segments == []


def test_tiers_empty_and_repr() -> None:
    empty = topology.RnaSecondaryStructureTopology("", "")
    assert empty.tiers == []
    assert "tiers=" in repr(empty)

    structure = topology.RnaSecondaryStructureTopology("ACGU", "(())")
    _ = structure.tiers
    assert "tiers=" in repr(structure)


def test_topology_empty_crossing_helpers_and_repr() -> None:
    empty = topology.RnaSecondaryStructureTopology("", "")
    assert empty.crossing_events.numel() == 0
    assert empty.crossing_arcs.numel() == 0
    assert empty.crossing_pairs.numel() == 0
    assert empty.crossing_nucleotides.numel() == 0
    with pytest.raises(ValueError):
        empty.pairs(view="bad")

    simple = topology.RnaSecondaryStructureTopology("AAAA", "(())")
    assert "Tier" in repr(simple.tiers[0])

    crossing_loops = topology.Loops.from_pairs([(0, 3), (1, 4)], length=6)
    assert "pk=" in repr(crossing_loops)


def test_tier_at_out_of_range() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    with pytest.raises(IndexError):
        structure.tier_at(1)
    with pytest.raises(IndexError):
        structure.tier_at(-1)


def test_loop_segments_end_sides() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    loop_segments = structure.nested_loop_segments
    end_segments = [segment for segment in loop_segments.segments if segment.kind == topology.LoopSegmentType.END]
    assert len(end_segments) == 1
    assert end_segments[0].side == topology.EndSide.THREE_PRIME
    assert (end_segments[0].start, end_segments[0].stop) == (5, 5)

    structure = topology.RnaSecondaryStructureTopology("ACGUAA", ".(...)")
    loop_segments = structure.nested_loop_segments
    end_segments = [segment for segment in loop_segments.segments if segment.kind == topology.LoopSegmentType.END]
    assert len(end_segments) == 1
    assert end_segments[0].side == topology.EndSide.FIVE_PRIME
    assert (end_segments[0].start, end_segments[0].stop) == (0, 0)


def test_loop_segments_positions_and_end_filters() -> None:
    structure = topology.RnaSecondaryStructureTopology("A" * 7, ".(...).")
    loop_segments = structure.nested_loop_segments

    end_5p = loop_segments.end_5p
    end_3p = loop_segments.end_3p
    assert end_5p is not None
    assert end_5p.side == topology.EndSide.FIVE_PRIME
    assert (end_5p.start, end_5p.stop) == (0, 0)
    assert end_3p is not None
    assert end_3p.side == topology.EndSide.THREE_PRIME
    assert (end_3p.start, end_3p.stop) == (6, 6)

    end_positions = loop_segments.positions(kind=topology.LoopSegmentType.END)
    assert len(end_positions) == 2
    end_5p_positions = loop_segments.positions(kind=topology.LoopSegmentType.END, side=topology.EndSide.FIVE_PRIME)
    end_3p_positions = loop_segments.positions(kind=topology.LoopSegmentType.END, side=topology.EndSide.THREE_PRIME)
    assert len(end_5p_positions) == 1
    assert as_list(end_5p_positions[0]) == [0]
    assert len(end_3p_positions) == 1
    assert as_list(end_3p_positions[0]) == [6]

    hairpin_positions = loop_segments.positions(kind=topology.LoopSegmentType.HAIRPIN)
    assert hairpin_positions
    assert as_list(hairpin_positions[0]) == [2, 3, 4]

    all_5p_positions = loop_segments.positions(side=topology.EndSide.FIVE_PRIME)
    assert len(all_5p_positions) == 1
    assert as_list(all_5p_positions[0]) == [0]


def test_loop_segments_view_tiers() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    nested_loop_segments = structure.nested_loop_segments
    assert isinstance(nested_loop_segments, topology.LoopSegments)
    all_tiers = [tier.loop_segments for tier in structure.tiers]
    assert len(all_tiers) == len(structure.tiers)
    pk_tiers = [tier.loop_segments for tier in structure.tiers[1:]]
    assert len(pk_tiers) == len(structure.tiers[1:])
    assert all(tier.tier > 0 for tier in pk_tiers)


def test_loop_tiers_empty_pairs() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    tiers = structure.tiers
    assert len(tiers) == 1
    assert tiers[0].level == 0
    assert tiers[0].pairs.numel() == 0
    end_segments = [
        segment for segment in tiers[0].loop_segments.segments if segment.kind == topology.LoopSegmentType.END
    ]
    assert len(end_segments) == 1
    assert (end_segments[0].start, end_segments[0].stop) == (0, 3)
    assert end_segments[0].side is None


def test_loop_contexts_no_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    contexts = structure.tiers[0].loop_segments.contexts(structure.pseudoknot_pairs)
    assert len(contexts) == 1
    context = contexts[0]
    assert context.segment.tier == 0
    assert context.segment.kind == topology.LoopSegmentType.END
    assert context.pseudoknot_inside.numel() == 0
    assert context.pseudoknot_crossing.numel() == 0


def test_nested_loop_contexts_no_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    contexts = structure.loop_contexts()
    assert len(contexts) == 1
    context = contexts[0]
    assert context.loop.kind == topology.LoopType.EXTERNAL
    assert context.pseudoknot_inside.numel() == 0
    assert context.pseudoknot_crossing.numel() == 0


def test_loop_segment_contexts_no_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    contexts = structure.loop_segment_contexts()
    direct = structure.nested_loop_segments.contexts(structure.pseudoknot_pairs)
    assert len(contexts) == len(direct)
    for ctx, baseline in zip(contexts, direct):
        assert ctx.segment == baseline.segment
        assert ctx.pseudoknot_inside.numel() == 0
        assert ctx.pseudoknot_crossing.numel() == 0


def test_loop_segment_contexts_by_level_and_view() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    by_level = structure.loop_segment_contexts(level=1)
    expected = structure.tier_at(1).loop_segments.contexts(structure.pseudoknot_pairs)
    assert len(by_level) == len(expected)

    by_view = structure.loop_segment_contexts(view="nested")
    expected_view = structure.nested_loop_segments.contexts(structure.pseudoknot_pairs)
    assert len(by_view) == len(expected_view)


def test_structure_pseudoknot_edges_and_crossings() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")

    assert as_tuple_list(structure.nested_pairs) == [(1, 3)]
    assert as_tuple_list(structure.pseudoknot_pairs) == [(0, 2)]
    assert as_tuple_list(structure.crossing_pairs) == [(0, 2), (1, 3)]
    assert as_list(structure.crossing_arcs) == [[[0, 2], [1, 3]]]
    assert as_list(structure.crossing_events) == [[[0, 0, 2, 2], [1, 1, 3, 3]]]
    assert set(as_list(structure.crossing_nucleotides)) == {0, 1, 2, 3}

    edge_types = structure.edge_features["type"].reshape(-1)
    assert int((edge_types == topology.EdgeType.BACKBONE.value).sum().item()) == 3
    assert int((edge_types == topology.EdgeType.NESTED_PAIRS.value).sum().item()) == 1
    assert int((edge_types == topology.EdgeType.PSEUDOKNOT_PAIR.value).sum().item()) == 1


def test_loop_contexts_pseudoknot_crossing() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    tier_idx = next(
        idx
        for idx, tier in enumerate(structure.tiers)
        if set(as_tuple_list(tier.pairs)) == set(as_tuple_list(structure.nested_pairs))
    )
    contexts = structure.tiers[tier_idx].loop_segments.contexts(structure.pseudoknot_pairs)
    assert all(context.pseudoknot_inside.numel() == 0 for context in contexts)
    assert any(as_tuple_list(context.pseudoknot_crossing) == [(0, 2)] for context in contexts)


def test_nested_loop_contexts_pseudoknot_crossing() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    contexts = structure.loop_contexts()
    assert len(contexts) == len(structure.nested_loops)
    assert all(context.pseudoknot_inside.numel() == 0 for context in contexts)
    assert any(as_tuple_list(context.pseudoknot_crossing) == [(0, 2)] for context in contexts)


def test_loop_segment_contexts_by_tier() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    contexts_by_tier = [tier.loop_segments.contexts(structure.pseudoknot_pairs) for tier in structure.tiers]
    assert len(contexts_by_tier) == len(structure.tiers)
    tiers = {context.segment.tier for contexts in contexts_by_tier for context in contexts}
    assert tiers == {0, 1}


def test_loop_tiers_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    tiers = structure.tiers
    assert len(tiers) == 2
    assert as_tuple_list(tiers[0].pairs) == as_tuple_list(structure.nested_pairs)
    assert as_tuple_list(tiers[1].pairs) == as_tuple_list(structure.pseudoknot_pairs)


def test_loop_segments_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    loop_segments = [segment for tier in structure.tiers for segment in tier.loop_segments.segments]
    assert {segment.tier for segment in loop_segments} == {0, 1}
    hairpin_loops = [segment for segment in loop_segments if segment.kind == topology.LoopSegmentType.HAIRPIN]
    assert {segment.tier for segment in hairpin_loops} == {0, 1}
    ends = [segment for segment in loop_segments if segment.kind == topology.LoopSegmentType.END]
    assert {segment.tier for segment in ends} == {0, 1}


def test_stem_segments_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    stem_segments = [segment for tier in structure.tiers for segment in tier.stem_segments.segments]
    assert {segment.tier for segment in stem_segments} == {0, 1}
    assert len(stem_segments) == 2


def test_loops_no_pairs() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    loops = structure.loops()
    external_loops = [loop for loop in loops if loop.kind == topology.LoopType.EXTERNAL]
    assert len(external_loops) == 1
    loop = external_loops[0]
    assert loop.kind == topology.LoopType.EXTERNAL
    assert _loop_spans(loop) == [(0, 3)]
    assert loop.size == 4
    spans = loops.spans()
    assert spans
    assert all(isinstance(span, topology.LoopSpan) for span in spans)
    sliced = loops[:1]
    assert isinstance(sliced, topology.Loops)
    assert len(sliced) == 1


def test_loops_hairpin_external() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    loops = structure.loops()
    hairpin_loops = [loop for loop in loops if loop.kind == topology.LoopType.HAIRPIN]
    external_loops = [loop for loop in loops if loop.kind == topology.LoopType.EXTERNAL]
    assert len(hairpin_loops) == 1
    assert len(external_loops) == 1
    assert _loop_spans(hairpin_loops[0]) == [(1, 3)]
    assert _loop_spans(external_loops[0]) == [(5, 5)]


def test_loops_loop_at() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    loops = structure.loops()
    assert loops.loop_at(2) is not None
    assert loops.loop_at(2).kind == topology.LoopType.HAIRPIN
    assert loops.loop_at(5) is not None
    assert loops.loop_at(5).kind == topology.LoopType.EXTERNAL
    assert loops.loop_at(0) is None
    with pytest.raises(IndexError):
        loops.loop_at(len(structure))


def test_view_api_pairs_stems_loops() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    assert as_tuple_list(structure.pairs(view="all")) == [(0, 2), (1, 3)]
    assert as_tuple_list(structure.pairs(view="nested")) == [(1, 3)]
    assert as_tuple_list(structure.pairs(view="pseudoknot")) == [(0, 2)]

    stems_all = structure.stems(view="all")
    assert {stem.tier for stem in stems_all} == {0, 1}
    stems_nested = structure.stems(view="nested")
    assert {stem.tier for stem in stems_nested} == {0}
    stems_pseudoknot = structure.stems(view="pseudoknot")
    assert {stem.tier for stem in stems_pseudoknot} == {1}

    helices_all = structure.helices(view="all")
    assert {helix.tier for helix in helices_all} == {0, 1}
    helices_nested = structure.helices(view="nested")
    assert {helix.tier for helix in helices_nested} == {0}
    helices_pseudoknot = structure.helices(view="pseudoknot")
    assert {helix.tier for helix in helices_pseudoknot} == {1}

    loops_all = structure.loops(view="all")
    loops_nested = structure.loops(view="nested")
    loops_pseudoknot = structure.loops(view="pseudoknot")
    loops_nested_positional = structure.loops("nested")
    assert _loop_signature(loops_all) == _loop_signature(topology.Loops.from_pairs(structure.all_pairs, len(structure)))
    assert _loop_signature(loops_nested) == _loop_signature(
        topology.Loops.from_pairs(structure.nested_pairs, len(structure))
    )
    assert _loop_signature(loops_pseudoknot) == _loop_signature(
        topology.Loops.from_pairs(structure.pseudoknot_pairs, len(structure))
    )
    assert _loop_signature(loops_nested_positional) == _loop_signature(loops_nested)


def test_residue_lookup_helpers_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    assert structure.partner_at(1) == 3
    assert structure.partner_at(0) == 2
    assert structure.partner_at(0, level=0) is None
    assert structure.partner_at(0, level=1) == 2

    stem = structure.stem_at(1)
    assert stem is not None
    assert stem.tier == 0
    assert structure.stem_at(0) is not None
    assert structure.stem_at(0).tier == 1
    assert structure.stem_at(0, level=0) is None

    helix = structure.helix_at(1)
    assert helix is not None
    assert helix.tier == 0
    assert structure.helix_at(0) is not None
    assert structure.helix_at(0).tier == 1


def test_residue_lookup_helpers_view() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    assert structure.partner_at(1, view="all") == 3
    assert structure.partner_at(1, view="nested") == 3
    assert structure.partner_at(1, view="pseudoknot") is None
    assert structure.partner_at(0, view="pseudoknot") == 2
    assert structure.partner_at(0, view="nested") is None

    stem = structure.stem_at(1, view="nested")
    assert stem is not None
    assert stem.tier == 0
    assert structure.stem_at(1, view="pseudoknot") is None
    stem_pk = structure.stem_at(0, view="pseudoknot")
    assert stem_pk is not None
    assert stem_pk.tier == 1

    helix = structure.helix_at(1, view="nested")
    assert helix is not None
    assert helix.tier == 0
    assert structure.helix_at(1, view="pseudoknot") is None
    helix_pk = structure.helix_at(0, view="pseudoknot")
    assert helix_pk is not None
    assert helix_pk.tier == 1

    loop = structure.loop_at(1, view="pseudoknot")
    assert isinstance(loop, topology.Loop)
    assert loop.kind == topology.LoopType.HAIRPIN
    assert structure.loop_at(1, view="nested") is None


def test_paired_unpaired_positions_views() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    assert structure.paired_positions() == [0, 1, 2, 3]
    assert structure.unpaired_positions() == []
    assert structure.paired_positions(view="nested") == [1, 3]
    assert structure.unpaired_positions(view="nested") == [0, 2]
    assert structure.paired_positions(view="pseudoknot") == [0, 2]
    assert structure.unpaired_positions(view="pseudoknot") == [1, 3]

    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    assert structure.paired_positions() == []
    assert structure.unpaired_positions() == [0, 1, 2, 3]


def test_noncanonical_pairs_view() -> None:
    pairs = torch.tensor([[0, 3]])
    structure = topology.RnaSecondaryStructureTopology("ACGA", pairs)
    assert as_tuple_list(structure.noncanonical_pairs()) == [(0, 3)]
    assert as_tuple_list(structure.noncanonical_pairs(view="nested")) == [(0, 3)]
    assert as_tuple_list(structure.noncanonical_pairs(view="pseudoknot")) == []

    structure = topology.RnaSecondaryStructureTopology("ACGU", pairs)
    assert structure.noncanonical_pairs().numel() == 0


def test_annotate_positions_basic() -> None:
    structure = topology.RnaSecondaryStructureTopology("A" * 5, "([)].")
    annotations = structure.annotate_positions()
    assert annotations[1]["paired"] is True
    assert annotations[1]["segment_type"] == "stem"
    assert isinstance(annotations[1]["segment"], topology.StemSegment)
    assert annotations[1]["tier"] == 0
    assert annotations[4]["paired"] is False
    assert isinstance(annotations[4]["loop_segment"], topology.LoopSegment)
    assert annotations[4]["loop_kind"] == topology.LoopType.EXTERNAL

    helix_annotations = structure.annotate_positions(paired="helix")
    assert helix_annotations[1]["segment_type"] == "helix"

    pref_annotations = structure.annotate_positions(tier_preference="pseudoknot")
    assert pref_annotations[4]["tier"] == 1


def test_residue_lookup_helpers_unpaired() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    assert structure.partner_at(0) is None
    assert structure.stem_at(0) is None
    assert structure.helix_at(0) is None


def test_region_at_basic() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    region = structure.region_at(0)
    assert isinstance(region, topology.StemSegment)
    assert region.tier == 0
    region = structure.region_at(0, paired="helix")
    assert isinstance(region, topology.HelixSegment)

    region = structure.region_at(2)
    assert isinstance(region, topology.LoopSegment)
    assert region.kind == topology.LoopSegmentType.HAIRPIN

    region = structure.region_at(5)
    assert isinstance(region, topology.LoopSegment)
    assert region.kind == topology.LoopSegmentType.END


def test_region_at_unpaired_loop() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    region = structure.region_at(2, unpaired="loop")
    assert isinstance(region, topology.Loop)
    assert region.kind == topology.LoopType.HAIRPIN
    region = structure.region_at(5, unpaired="loop")
    assert isinstance(region, topology.Loop)
    assert region.kind == topology.LoopType.EXTERNAL
    region = structure.region_at(0, unpaired="loop")
    assert isinstance(region, topology.StemSegment)


def test_region_at_view_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    region = structure.region_at(0, view="pseudoknot")
    assert isinstance(region, topology.StemSegment)
    assert region.tier == 1

    region = structure.region_at(0, view="nested")
    assert isinstance(region, topology.LoopSegment)
    assert region.kind == topology.LoopSegmentType.END


def test_region_at_view_pseudoknot_unpaired() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    region = structure.region_at(1, view="pseudoknot")
    assert isinstance(region, topology.LoopSegment)
    assert region.kind == topology.LoopSegmentType.HAIRPIN
    assert region.tier == 1
    loop = structure.region_at(1, view="pseudoknot", unpaired="loop")
    assert isinstance(loop, topology.Loop)
    assert loop.kind == topology.LoopType.HAIRPIN


def test_region_at_prefers_paired() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    region = structure.region_at(0)
    assert isinstance(region, topology.StemSegment)
    assert region.tier == 1


def test_region_and_loop_segment_tier_preference() -> None:
    structure = topology.RnaSecondaryStructureTopology("A" * 5, "([)].")

    segment_nested = structure.loop_segment_at(4, view="all", tier_preference="nested")
    segment_pk = structure.loop_segment_at(4, view="all", tier_preference="pseudoknot")
    segment_last = structure.loop_segment_at(4, view="all", tier_preference="last")
    assert segment_nested is not None
    assert segment_nested.tier == 0
    assert segment_pk is not None
    assert segment_pk.tier == 1
    assert segment_last is not None
    assert segment_last.tier == 1

    region = structure.region_at(4, view="all")
    assert isinstance(region, topology.LoopSegment)
    assert region.tier == 0

    pk_region = structure.region_at(1, view="all", tier_preference="pseudoknot")
    assert isinstance(pk_region, topology.LoopSegment)
    assert pk_region.tier == 1

    nested_loop_region = structure.region_at(1, unpaired="loop", tier_preference="nested")
    assert isinstance(nested_loop_region, topology.StemSegment)
    pk_loop_region = structure.region_at(1, unpaired="loop", tier_preference="pseudoknot")
    assert isinstance(pk_loop_region, topology.Loop)
    assert pk_loop_region.kind == topology.LoopType.HAIRPIN


def test_loop_segment_at_view() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    segment = structure.loop_segment_at(0, view="nested")
    assert isinstance(segment, topology.LoopSegment)
    assert segment.kind == topology.LoopSegmentType.END
    assert segment.tier == 0
    assert structure.loop_segment_at(0, view="pseudoknot") is None

    segment = structure.loop_segment_at(1, view="pseudoknot")
    assert isinstance(segment, topology.LoopSegment)
    assert segment.kind == topology.LoopSegmentType.HAIRPIN
    assert segment.tier == 1
    assert structure.loop_segment_at(1, view="nested") is None


def test_regions_at_view_all() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    regions = structure.regions_at(1, view="all")
    assert len(regions) == 2
    assert isinstance(regions[0], topology.StemSegment)
    assert regions[0].tier == 0
    assert isinstance(regions[1], topology.LoopSegment)
    assert regions[1].tier == 1

    regions = structure.regions_at(1, view="nested")
    assert len(regions) == 1
    assert isinstance(regions[0], topology.StemSegment)
    assert regions[0].tier == 0

    regions = structure.regions_at(1, view="pseudoknot")
    assert len(regions) == 1
    assert isinstance(regions[0], topology.LoopSegment)
    assert regions[0].tier == 1

    regions = structure.regions_at(1, view="nested", paired="helix")
    assert len(regions) == 1
    assert isinstance(regions[0], topology.HelixSegment)
    assert regions[0].tier == 0


def test_loops_internal_loop() -> None:
    structure = topology.RnaSecondaryStructureTopology("AAAAAA", "(.(.))")
    loops = structure.loops()
    internal_loops = [loop for loop in loops if loop.kind == topology.LoopType.INTERNAL]
    assert len(internal_loops) == 1
    assert _loop_spans(internal_loops[0]) == [(1, 1), (3, 3)]


def test_loops_mode_nested_alias() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    nested_view = structure.loops(view="nested")
    nested_mode = structure.loops(mode="nested")
    assert _loop_signature(nested_view) == _loop_signature(nested_mode)


def test_loops_mode_nested_default_view() -> None:
    structure = topology.RnaSecondaryStructureTopology("A" * 5, "([)].")
    loops_mode = structure.loops(mode="nested")
    loops_view = structure.loops(view="nested")
    assert _loop_signature(loops_mode) == _loop_signature(loops_view)


def test_loops_bulge() -> None:
    structure = topology.RnaSecondaryStructureTopology("AAAAA", "(.())")
    loops = structure.loops()
    bulge_loops = [loop for loop in loops if loop.kind == topology.LoopType.BULGE]
    assert len(bulge_loops) == 1
    assert _loop_spans(bulge_loops[0]) == [(1, 1)]


def test_loops_multiloop() -> None:
    pairs = [(0, 11), (2, 3), (5, 6), (8, 9)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=12)
    structure = topology.RnaSecondaryStructureTopology("A" * 12, dot_bracket)
    loops = structure.loops()
    multi_loops = [loop for loop in loops if loop.kind == topology.LoopType.MULTILOOP]
    assert len(multi_loops) == 1
    assert _loop_spans(multi_loops[0]) == [(1, 1), (4, 4), (7, 7), (10, 10)]


def test_loops_pseudoknot_segments() -> None:
    pairs = [(0, 3), (1, 4)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=6)
    structure = topology.RnaSecondaryStructureTopology("A" * 6, dot_bracket)
    loops = structure.loops()
    nested_loops = structure.loops(view="nested")
    assert _loop_signature(loops) == _loop_signature(topology.Loops.from_pairs(structure.all_pairs, len(structure)))
    external_loops = [loop for loop in loops if loop.kind == topology.LoopType.EXTERNAL]
    assert len(external_loops) == 1
    assert any(span.stop == len(structure) - 1 for span in external_loops[0].spans)
    assert _loop_signature(nested_loops) == _loop_signature(
        topology.Loops.from_pairs(structure.nested_pairs, len(structure))
    )
    assert any(loop.kind == topology.LoopType.HAIRPIN for loop in nested_loops)


def test_loops_taxonomy_h_type_roles() -> None:
    pairs = [(0, 9), (1, 8), (3, 12), (4, 11)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=13)
    structure = topology.RnaSecondaryStructureTopology("A" * 13, dot_bracket)
    loops = structure.loops(mode=topology.LoopView.Taxonomy)
    roles = {loop.role: _loop_spans(loop) for loop in loops if loop.role is not None}
    assert roles[topology.LoopRole.L1] == [(2, 2)]
    assert roles[topology.LoopRole.L2] == [(5, 7)]
    assert roles[topology.LoopRole.L3] == [(10, 10)]
    assert all(loop.is_nested for loop in loops if loop.role is not None)
    assert all(loop.pseudoknot_type == topology.PseudoknotType.H_TYPE for loop in loops if loop.role is not None)


def test_loops_taxonomy_kissing_hairpin_roles() -> None:
    pairs = [(0, 7), (1, 6), (9, 15), (10, 14), (3, 12), (4, 11)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=16)
    structure = topology.RnaSecondaryStructureTopology("A" * 16, dot_bracket)
    loops = structure.loops(mode=topology.LoopView.Taxonomy)
    roles = {loop.role: _loop_spans(loop) for loop in loops if loop.role is not None}
    assert roles[topology.LoopRole.K1] == [(2, 5)]
    assert roles[topology.LoopRole.K2] == [(11, 13)]
    assert all(
        loop.pseudoknot_type == topology.PseudoknotType.KISSING_HAIRPIN for loop in loops if loop.role is not None
    )


def test_loops_taxonomy_kissing_hairpin_complex() -> None:
    pairs = [(0, 15), (1, 14), (16, 31), (17, 30), (3, 28), (4, 27), (8, 23), (9, 22)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=32)
    structure = topology.RnaSecondaryStructureTopology("A" * 32, dot_bracket)
    loops = structure.loops(mode="taxonomy")
    assert any(loop.pseudoknot_type == topology.PseudoknotType.COMPLEX for loop in loops)
    assert all(loop.pseudoknot_type != topology.PseudoknotType.KISSING_HAIRPIN for loop in loops)


def test_loops_taxonomy_kissing_hairpin_bulge_merge() -> None:
    pairs = [(0, 7), (1, 6), (9, 15), (10, 14), (3, 12), (5, 11)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=16)
    structure = topology.RnaSecondaryStructureTopology("A" * 16, dot_bracket)
    loops = structure.loops(mode="taxonomy")
    assert any(loop.pseudoknot_type == topology.PseudoknotType.KISSING_HAIRPIN for loop in loops)


def test_loops_taxonomy_unknown_pseudoknot() -> None:
    pairs = [(0, 7), (1, 6), (2, 9), (3, 8), (4, 11), (5, 10)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=12)
    structure = topology.RnaSecondaryStructureTopology("A" * 12, dot_bracket)
    loops = structure.loops(mode="taxonomy")
    assert all(loop.pseudoknot_type == topology.PseudoknotType.UNKNOWN for loop in loops)
    assert all(loop.is_nested is False for loop in loops)


def test_loops_taxonomy_m_type_classification() -> None:
    pairs = [(0, 11), (1, 10), (3, 14), (4, 13), (6, 17), (7, 16)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=18)
    structure = topology.RnaSecondaryStructureTopology("A" * 18, dot_bracket)
    loops = structure.loops(mode="taxonomy")
    assert all(loop.pseudoknot_type == topology.PseudoknotType.M_TYPE for loop in loops)
    assert all(loop.is_nested is False for loop in loops)


def test_loops_taxonomy_m_type_loop_level() -> None:
    pairs = [(2, 13), (3, 12), (5, 16), (6, 15), (8, 19), (9, 18)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=22)
    structure = topology.RnaSecondaryStructureTopology("A" * 22, dot_bracket)
    loops = structure.loops(mode="taxonomy")
    pk_loops = [loop for loop in loops if loop.pseudoknot_type == topology.PseudoknotType.M_TYPE]
    non_pk_loops = [loop for loop in loops if loop.pseudoknot_type == topology.PseudoknotType.NONE]
    assert pk_loops
    assert non_pk_loops
    assert all(loop.is_nested is False for loop in pk_loops)
    assert all(loop.is_nested is True for loop in non_pk_loops)


def test_taxonomy_loops_property() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    prop = structure.taxonomy_loops
    direct = structure.loops(mode="taxonomy")
    assert _loop_signature(prop) == _loop_signature(direct)
    assert all(loop.pseudoknot_type == topology.PseudoknotType.NONE for loop in prop)
    assert all(loop.is_nested for loop in prop)


def test_loop_sequences_hairpin() -> None:
    structure = topology.RnaSecondaryStructureTopology("AGCUGA", "(...).")
    loops = structure.loops()
    hairpin = [loop for loop in loops if loop.kind == topology.LoopType.HAIRPIN][0]
    assert structure.loop_span_sequences(hairpin) == ("GCU",)
    assert structure.loop_sequence(hairpin) == "GCU"
    assert structure.loop_sequence(hairpin, joiner="&") == "GCU"
    assert structure.loop_anchor_pair_sequences(hairpin) == ("AG",)


def test_loop_sequences_internal() -> None:
    structure = topology.RnaSecondaryStructureTopology("AGCUAG", "(.(.))")
    loops = structure.loops()
    internal = [loop for loop in loops if loop.kind == topology.LoopType.INTERNAL][0]
    assert structure.loop_span_sequences(internal) == ("G", "U")
    assert structure.loop_sequence(internal) == "GU"
    assert structure.loop_sequence(internal, joiner="&") == "G&U"
    assert structure.loop_anchor_pair_sequences(internal) == ("AG", "CA")


def test_stem_helix_strands() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUGC", "((..))")
    stem = structure.stems()[0]
    helix = structure.helices()[0]
    assert structure.stem_strands(stem) == ("AC", "CG")
    assert structure.helix_strands(helix) == ("AC", "CG")
    assert structure.stem_strands(stem, orientation="index") == ("AC", "GC")
    assert structure.helix_strands(helix, orientation="index") == ("AC", "GC")


def test_loop_helix_graph_hairpin() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGUAA", "(...).")
    loop_helix = structure.loop_helix_graph()
    assert len(loop_helix.loops) == 2
    assert len(loop_helix.helices) == 1
    assert loop_helix.loop_nodes == (0, 1)
    assert loop_helix.helix_nodes == (2,)
    assert set(as_tuple_list(loop_helix.graph.edge_index)) == {(0, 2), (1, 2)}
    tiers = loop_helix.graph.edge_features["tier"].reshape(-1).tolist()
    assert set(tiers) == {0}
    assert loop_helix.loop_helix_indices(0) == (0,)
    assert loop_helix.loop_helix_indices(1) == (0,)
    assert loop_helix.loop_neighbor_nodes(0) == (2,)
    assert loop_helix.loop_neighbor_nodes(1) == (2,)
    assert set(loop_helix.helix_loop_indices(0)) == {0, 1}
    assert set(loop_helix.helix_neighbor_nodes(0)) == {0, 1}
    assert len(loop_helix.loop_helices(0)) == 1
    assert len(loop_helix.helix_loops(0)) == 2
    assert set(loop_helix.loop_helix_edges()) == {(0, 0), (1, 0)}
    assert set(loop_helix.edge_list()) == {(0, 2), (1, 2)}
    adj = loop_helix.adjacency_matrix()
    assert tuple(adj.shape) == (3, 3)
    assert adj[0, 2].item() is True
    assert adj[2, 0].item() is True
    assert adj[1, 2].item() is True
    assert adj[2, 1].item() is True
    assert adj[0, 1].item() is False


def test_loop_helix_graph_multi_helix_neighbors() -> None:
    pairs = [(0, 11), (2, 3), (5, 6), (8, 9)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=12)
    structure = topology.RnaSecondaryStructureTopology("A" * 12, dot_bracket)
    loop_helix = structure.loop_helix_graph()
    assert len(loop_helix.helices) == 4
    assert len(loop_helix.loops) >= 1

    loop_count = len(loop_helix.loops)
    helix_count = len(loop_helix.helices)
    assert max(len(loop_helix.loop_helix_indices(idx)) for idx in range(loop_count)) > 1
    for idx in range(loop_count):
        neighbors = loop_helix.loop_neighbor_nodes(idx)
        assert all(node >= loop_count for node in neighbors)
        assert len(loop_helix.loop_helix_indices(idx)) == len(neighbors)
    for idx in range(helix_count):
        neighbors = loop_helix.helix_neighbor_nodes(idx)
        assert neighbors
        assert all(node < loop_count for node in neighbors)
        assert len(loop_helix.helix_loop_indices(idx)) == len(neighbors)


def test_loop_helix_graph_pairs_taxonomy() -> None:
    pairs = [(0, 5), (1, 4)]
    structure = topology.RnaSecondaryStructureTopology("A" * 6, "((..))")
    loop_helix = structure.loop_helix_graph(pairs=pairs, mode="taxonomy")
    assert len(loop_helix.helices) == 1
    assert len(loop_helix.loops) == 1
    assert loop_helix.loop_helix_edges() == [(0, 0)]


def test_loop_helix_graph_nested_view() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    loop_helix = structure.loop_helix_graph(view=topology.StructureView.NESTED)
    assert len(loop_helix.helices) == 1
    assert {helix.tier for helix in loop_helix.helices} == {0}
    assert len(loop_helix.graph.edge_index) == 2
    tiers = loop_helix.graph.edge_features["tier"].reshape(-1).tolist()
    assert set(tiers) == {0}


def test_loop_helix_graph_tier_filter() -> None:
    pairs = [(0, 7), (1, 6), (9, 15), (10, 14), (3, 12), (4, 11)]
    dot_bracket = pairs_to_dot_bracket(pairs, length=16)
    structure = topology.RnaSecondaryStructureTopology("A" * 16, dot_bracket)
    loop_helix = structure.loop_helix_graph(level=0)
    assert {helix.tier for helix in loop_helix.helices} == {0}
    tiers = loop_helix.graph.edge_features["tier"].reshape(-1).tolist()
    assert set(tiers) == {0}

    loop_helix = structure.loop_helix_graph(level=1)
    assert {helix.tier for helix in loop_helix.helices} == {1}
    tiers = loop_helix.graph.edge_features["tier"].reshape(-1).tolist()
    assert set(tiers) == {1}


def test_segment_edges_empty_pairs() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "....")
    assert structure.nested_stem_segments.edges == []
    assert structure.all_stem_segments.edges == []
    assert structure.tiers[0].stem_segments.edges == []
    assert structure.tiers[0].stem_segments.graph.num_nodes == 0


def test_segment_edges_pseudoknot() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACGU", "([)]")
    assert structure.nested_stem_segments.edges == []
    assert structure.all_stem_segments.edges != []
    assert [tier.stem_segments.edges for tier in structure.tiers] == [[], []]
    assert all(tier.stem_segments.edges == [] for tier in structure.tiers)
    assert all(tier.stem_segments.graph.num_nodes == 1 for tier in structure.tiers)


def test_stem_regions_contiguous() -> None:
    structure = topology.RnaSecondaryStructureTopology("AAAAAAA", "(((.)))")
    stem_segments = structure.nested_stem_segments.segments
    assert len(stem_segments) == 1
    segment = stem_segments[0]
    assert (segment.start_5p, segment.stop_5p) == (0, 2)
    assert (segment.start_3p, segment.stop_3p) == (6, 4)
    assert len(segment) == 3


def test_helix_segments_gap_split() -> None:
    pairs = torch.tensor([[0, 7], [1, 6], [3, 4]], dtype=torch.long)
    structure = topology.RnaSecondaryStructureTopology("ACGUACGU", pairs)
    stem_segments = structure.nested_stem_segments.segments
    assert len(stem_segments) == 1
    helix_segments = structure.nested_helix_segments.segments
    assert len(helix_segments) == 2
    segment = helix_segments[0]
    assert (segment.start_5p, segment.stop_5p) == (0, 1)
    assert (segment.start_3p, segment.stop_3p) == (7, 6)
    assert len(segment) == 2
    segment = helix_segments[1]
    assert (segment.start_5p, segment.stop_5p) == (3, 3)
    assert (segment.start_3p, segment.stop_3p) == (4, 4)
    assert len(segment) == 1


def test_open_to_close_basic() -> None:
    open_to_close = topology.LoopSegments.open_to_close(torch.tensor([[3, 1], [2, 4]]), 6)
    assert open_to_close[1] == 3
    assert open_to_close[2] == 4


@pytest.mark.parametrize(
    "pairs",
    [torch.tensor([1, 2, 3]), torch.tensor([[0, 1, 2]]), [1, 2, 3], [(0, 1, 2)]],
    ids=["torch_1d", "torch_2d_bad", "list_1d", "list_2d_bad"],
)
def test_open_to_close_shape_errors(pairs) -> None:
    with pytest.raises(ValueError, match="shape"):
        topology.LoopSegments.open_to_close(pairs, 4)


def test_open_to_close_empty_pairs() -> None:
    empty = topology.LoopSegments.open_to_close([], 0)
    assert empty.numel() == 0
    nonempty = topology.LoopSegments.open_to_close([], 3)
    assert as_list(nonempty) == [-1, -1, -1]


def test_secondary_structure_device_and_crossing_pairs() -> None:
    structure = topology.RnaSecondaryStructureTopology("AC", "..", device=torch.device("cpu"))
    assert structure.pairs().device.type == "cpu"
    assert structure.crossing_pairs.numel() == 0

    single_pair = topology.RnaSecondaryStructureTopology("ACG", "(.)")
    assert as_tuple_list(single_pair.crossing_pairs) == []


def test_stem_regions_single_pair() -> None:
    structure = topology.RnaSecondaryStructureTopology("ACG", "(.)")
    stem_segments = structure.nested_stem_segments.segments
    assert len(stem_segments) == 1
    segment = stem_segments[0]
    assert (segment.start_5p, segment.stop_5p) == (0, 0)
    assert (segment.start_3p, segment.stop_3p) == (2, 2)
    assert len(segment) == 1


@pytest.mark.parametrize(
    "dot_bracket, expected_pairs, expected_primary, expected_pseudoknot, expected_open",
    [
        (
            "((..))",
            [(0, 5), (1, 4)],
            [(0, 5), (1, 4)],
            [],
            [5, 4, -1, -1, -1, -1],
        ),
        (
            "([)]",
            [(0, 2), (1, 3)],
            [(1, 3)],
            [(0, 2)],
            [-1, 3, -1, -1],
        ),
    ],
    ids=["nested", "pseudoknot"],
)
def test_dot_bracket_paths(
    dot_bracket: str,
    expected_pairs: list[tuple[int, int]],
    expected_primary: list[tuple[int, int]],
    expected_pseudoknot: list[tuple[int, int]],
    expected_open: list[int],
) -> None:
    pairs = topology.dot_bracket_to_pairs(dot_bracket)
    nested_pairs, pseudoknot_pairs = topology.split_pseudoknot_pairs(pairs)
    assert as_tuple_list(pairs) == expected_pairs
    assert as_tuple_list(nested_pairs) == expected_primary
    assert as_tuple_list(pseudoknot_pairs) == expected_pseudoknot
    nested_open = topology.LoopSegments.open_to_close(nested_pairs, len(dot_bracket))
    assert as_list(nested_open) == expected_open


def test_secondary_structure_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        topology.RnaSecondaryStructureTopology("AC", "(.)")


def test_topology_input_validation_errors() -> None:
    with pytest.raises(TypeError):
        topology.RnaSecondaryStructureTopology("AC", 123)

    structure = topology.RnaSecondaryStructureTopology("ACGU", "(())")

    with pytest.raises(ValueError):
        structure.loop_segment_contexts(view="nested", level=0)
    with pytest.raises(ValueError):
        structure.loop_segment_contexts(view="all")

    with pytest.raises(IndexError):
        structure.partner_at(10)
    with pytest.raises(ValueError):
        structure.partner_at(0, view="nested", level=0)

    with pytest.raises(ValueError):
        structure.stem_at(0, view="nested", level=0)
    with pytest.raises(ValueError):
        structure.helix_at(0, view="nested", level=0)

    with pytest.raises(ValueError):
        structure.region_at(0, unpaired="bad")
    with pytest.raises(ValueError):
        structure.region_at(0, unpaired="loop", paired="bad")

    with pytest.raises(ValueError):
        structure.regions_at(0, paired="bad")
    with pytest.raises(IndexError):
        structure.regions_at(10)

    with pytest.raises(ValueError):
        structure.annotate_positions(paired="bad")

    with pytest.raises(ValueError):
        structure.loops(view="nested", pairs=[(0, 3)])
    with pytest.raises(ValueError):
        structure.loops(view="pseudoknot", mode="nested")

    with pytest.raises(ValueError):
        structure.stem_strands(structure.stems()[0], orientation="bad")

    with pytest.raises(ValueError):
        topology.LoopSegments([(0, 2), (1, 3)], 4)


@pytest.mark.parametrize(
    "open_to_close, expected",
    [
        ([-1, -1, -1, -1, -1, -1], {topology.LoopSegmentType.END}),
        (
            [-1, 4, -1, -1, -1, -1],
            {topology.LoopSegmentType.HAIRPIN, topology.LoopSegmentType.END},
        ),
        (
            [6, -1, 5, -1, -1, -1, -1],
            {topology.LoopSegmentType.BULGE, topology.LoopSegmentType.HAIRPIN},
        ),
        (
            [5, 3, -1, -1, -1, -1],
            {topology.LoopSegmentType.BULGE, topology.LoopSegmentType.HAIRPIN},
        ),
        (
            [7, -1, 5, -1, -1, -1, -1, -1],
            {topology.LoopSegmentType.INTERNAL, topology.LoopSegmentType.HAIRPIN},
        ),
        (
            [9, -1, 4, -1, -1, -1, 8, -1, -1, -1],
            {topology.LoopSegmentType.BRANCH, topology.LoopSegmentType.HAIRPIN},
        ),
        (
            [-1, 4, -1, -1, -1, -1, 9, -1, -1, -1, -1],
            {topology.LoopSegmentType.EXTERNAL, topology.LoopSegmentType.END, topology.LoopSegmentType.HAIRPIN},
        ),
    ],
    ids=[
        "all_end",
        "hairpin_with_end",
        "bulge_left",
        "bulge_right",
        "internal",
        "branch",
        "external",
    ],
)
def test_loop_types(open_to_close: list[int], expected: Set[topology.LoopSegmentType]) -> None:
    pairs = [(idx, j) for idx, j in enumerate(open_to_close) if j != -1]
    loop_segments = topology.LoopSegments(pairs, len(open_to_close), torch.device("cpu"))
    assert _nonempty_loop_types(loop_segments) == expected


def test_loops_zero_length() -> None:
    loop_segments = topology.LoopSegments([], 0, torch.device("cpu"))
    assert loop_segments.segments == []
    assert loop_segments.segments == []


def test_loops_empty_intervals_error() -> None:
    loop_segments = topology.LoopSegments([(0, 1)], 2, torch.device("cpu"))
    assert loop_segments.segments == []
