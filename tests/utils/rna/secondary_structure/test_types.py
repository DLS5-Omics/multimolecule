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

import pytest
import torch

from multimolecule.utils.rna.secondary_structure.types import (
    EndSide,
    HelixSegment,
    Loop,
    LoopRole,
    LoopSegment,
    LoopSegmentContext,
    LoopSegmentType,
    LoopSpan,
    LoopType,
    LoopView,
    PseudoknotType,
    StemSegment,
    StructureView,
)


def test_types_loop_helpers_and_ordering() -> None:
    spans = (LoopSpan(2, 4), LoopSpan(10, 10))
    helix1 = HelixSegment(1, 2, 5, 4, ((1, 5), (2, 4)), 0)
    helix2 = HelixSegment(11, 11, 9, 9, ((11, 9),), 1)
    loop = Loop(
        LoopType.INTERNAL,
        spans,
        anchor_pairs=((1, 5), (11, 9)),
        anchor_helices=(helix1, helix2),
        anchor_tiers=(0, 1),
        is_external=False,
        role=LoopRole.L1,
    )
    assert loop.size == 4
    assert loop.span_lengths == (3, 1)
    assert loop.span_count == 2
    assert loop.anchor_helix_count == 2
    assert loop.branch_count == 2
    assert loop.asymmetry == 2
    assert loop.overlaps_span(3, 3)
    assert not loop.overlaps_span(6, 8)
    assert loop.anchor_positions() == (1, 9)
    assert loop.ordered_anchor_pairs() == ((1, 5), (11, 9))
    assert loop.ordered_anchor_helices()[0] == helix1
    updated = loop.with_taxonomy(PseudoknotType.H_TYPE, False)
    assert updated.pseudoknot_type == PseudoknotType.H_TYPE
    assert not updated.is_nested
    assert updated.role == LoopRole.L1

    empty_loop = Loop(
        LoopType.HAIRPIN,
        (LoopSpan(0, 0),),
        anchor_pairs=(),
        anchor_helices=(),
        anchor_tiers=(),
        is_external=False,
    )
    assert empty_loop.anchor_positions() == ()
    assert empty_loop.ordered_anchor_pairs() == ()
    assert empty_loop.ordered_anchor_helices() == ()
    assert empty_loop.asymmetry == 0

    helix_orphan = HelixSegment(8, 9, 3, 2, ((8, 3),), 0)
    orphan_loop = Loop(
        LoopType.BULGE,
        (LoopSpan(5, 6),),
        anchor_pairs=((1, 10),),
        anchor_helices=(helix_orphan,),
        anchor_tiers=(0,),
        is_external=False,
    )
    assert orphan_loop.ordered_anchor_helices()[0] == helix_orphan

    assert StructureView.parse(None) == StructureView.ALL
    assert StructureView.parse("nested") == StructureView.NESTED
    with pytest.raises(ValueError):
        StructureView.parse("unknown")
    assert LoopView.parse(LoopView.Taxonomy) == LoopView.Taxonomy
    assert LoopView.parse(LoopView.Topological.value) == LoopView.Topological

    segment = LoopSegment(LoopSegmentType.END, 5, 6, tier=1, side=EndSide.THREE_PRIME)
    assert len(segment) == 2
    assert LoopSegmentContext(segment, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    stem = StemSegment(0, 1, 6, 5, 0)
    assert stem.key == (0, 1, 6, 5)
    assert len(stem) == 2
    assert helix1.key == (1, 2, 5, 4)
