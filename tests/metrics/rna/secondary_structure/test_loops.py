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

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from multimolecule.metrics.rna.secondary_structure import loops
from multimolecule.metrics.rna.secondary_structure.common import f1_from_confusion, loop_taxonomy_compatible
from multimolecule.utils.rna.secondary_structure import (
    EndSide,
    LoopSegment,
    LoopSegmentType,
    LoopSpan,
    LoopType,
    PseudoknotType,
)
from tests.metrics.rna.secondary_structure.utils import assert_confusion_metric_family


@dataclass(frozen=True)
class _FakeHelix:
    key: tuple[int, int, int, int]


class _FakeLoop:
    def __init__(
        self,
        *,
        kind: LoopType = LoopType.HAIRPIN,
        anchor_pairs: list[tuple[int, int]] | None = None,
        ordered_pairs: list[tuple[int, int]] | None = None,
        anchor_helices: list[_FakeHelix] | None = None,
        ordered_helices: list[_FakeHelix] | None = None,
        pseudoknot_type: PseudoknotType = PseudoknotType.NONE,
        role: object | None = None,
        spans: list[LoopSpan] | None = None,
    ) -> None:
        self.kind = kind
        self.anchor_pairs = [] if anchor_pairs is None else list(anchor_pairs)
        self._ordered_pairs = self.anchor_pairs if ordered_pairs is None else list(ordered_pairs)
        self.anchor_helices = [] if anchor_helices is None else list(anchor_helices)
        self._ordered_helices = self.anchor_helices if ordered_helices is None else list(ordered_helices)
        self.pseudoknot_type = pseudoknot_type
        self.role = role
        self.spans = [LoopSpan(0, 2)] if spans is None else spans
        self.size = sum(span.stop - span.start + 1 for span in self.spans)

    def ordered_anchor_pairs(self):
        return tuple(self._ordered_pairs)

    def ordered_anchor_helices(self):
        return tuple(self._ordered_helices)


class TestLoopAnchors:
    def test_scores_taxonomy_and_anchor_pairs(self) -> None:
        pred = _FakeLoop(pseudoknot_type=PseudoknotType.H_TYPE, role="L1")
        tgt_unknown = _FakeLoop(pseudoknot_type=PseudoknotType.UNKNOWN, role="L1")
        tgt_other = _FakeLoop(pseudoknot_type=PseudoknotType.KISSING_HAIRPIN, role="L1")
        tgt_role = _FakeLoop(pseudoknot_type=PseudoknotType.H_TYPE, role="L2")

        assert loop_taxonomy_compatible(pred, tgt_unknown)
        assert not loop_taxonomy_compatible(pred, tgt_other)
        assert not loop_taxonomy_compatible(pred, tgt_role)

        pred_loop = _FakeLoop(
            kind=LoopType.MULTILOOP,
            anchor_pairs=[(0, 9), (2, 7), (4, 5)],
            ordered_pairs=[(0, 9), (2, 7), (4, 5)],
        )
        tgt_loop = _FakeLoop(
            kind=LoopType.MULTILOOP,
            anchor_pairs=[(2, 7), (4, 5), (0, 9)],
            ordered_pairs=[(2, 7), (4, 5), (0, 9)],
        )
        assert (
            loops._cyclic_anchor_pairs_score(pred_loop.ordered_anchor_pairs(), tgt_loop.ordered_anchor_pairs()) == 1.0
        )
        assert loops._loop_anchor_pairs_score(LoopType.MULTILOOP, pred_loop, tgt_loop) == 1.0
        assert loops.loop_anchor_pairs_score_any(pred_loop, tgt_loop) == 1.0

        mismatch = _FakeLoop(anchor_pairs=[(0, 9), (2, 7)])
        assert loops.loop_anchor_pairs_score_any(pred_loop, mismatch) == 0.0
        assert loops.loop_anchor_pairs_score_any(_FakeLoop(anchor_pairs=[]), _FakeLoop(anchor_pairs=[])) == 1.0

    def test_maps_anchor_ids_and_scores_order(self) -> None:
        assert loops._ordered_anchor_ids_score([], []) == 1.0
        assert loops._ordered_anchor_ids_score([1], []) == 0.0
        assert loops._ordered_anchor_ids_score([1, 2], [1, 2]) == 1.0
        assert loops._ordered_anchor_ids_score([1, -1, 3], [1, 2, 3]) == pytest.approx(2.0 / 3.0)

        assert loops._cyclic_anchor_ids_score([1, 2, 3], [2, 3, 1]) == 1.0
        assert loops._cyclic_anchor_ids_score([], []) == 1.0

        assert loops._unordered_anchor_ids_score([], []) == 1.0
        assert loops._unordered_anchor_ids_score([1, 1, 2], [1, 2, 2]) == pytest.approx(2.0 / 3.0)
        assert loops._unordered_anchor_ids_score([-1], [-1]) == 0.0

        pred_ids = [2, -1, 5]
        mapped = loops._map_anchor_ids(pred_ids, {2: 8, 5: 9})
        assert mapped == [8, -1, 9]

    def test_scores_wiring_and_edge_similarity(self) -> None:
        h1 = _FakeHelix((0, 1, 9, 8))
        h2 = _FakeHelix((2, 3, 7, 6))
        h3 = _FakeHelix((4, 5, 5, 4))
        pred = _FakeLoop(kind=LoopType.MULTILOOP, anchor_helices=[h1, h2, h3], ordered_helices=[h1, h2, h3])
        tgt = _FakeLoop(kind=LoopType.MULTILOOP, anchor_helices=[h2, h3, h1], ordered_helices=[h2, h3, h1])
        pred_key_map = {h1.key: 0, h2.key: 1, h3.key: 2}
        tgt_key_map = {h2.key: 10, h3.key: 11, h1.key: 12}
        pred_to_tgt = {0: 12, 1: 10, 2: 11}

        wiring = loops.loop_anchor_wiring_score(LoopType.MULTILOOP, pred, tgt, pred_key_map, tgt_key_map, pred_to_tgt)
        assert wiring == pytest.approx(1.0)

        pred_edges = [(0, 1, 1), (1, 2, 2), (2, 0, 3)]
        tgt_edges = [(12, 10, 1), (10, 11, 2), (11, 12, 3)]
        adj_score = loops.loop_anchor_adjacency_score(
            pred, tgt, pred_key_map, tgt_key_map, pred_to_tgt, pred_edges, tgt_edges
        )
        assert adj_score == 1.0
        assert (
            loops._loop_anchor_order_score(LoopType.HAIRPIN, pred, tgt, pred_key_map, tgt_key_map, pred_to_tgt) == 1.0
        )
        assert (
            loops._loop_anchor_order_score(LoopType.MULTILOOP, pred, tgt, pred_key_map, tgt_key_map, pred_to_tgt) == 1.0
        )
        assert loops.loop_anchor_order_score_any(pred, tgt, pred_key_map, tgt_key_map, pred_to_tgt) == 1.0

        coax_pred = loops._loop_anchor_coaxial_edges(pred, pred_key_map, pred_edges)
        coax_tgt = loops._loop_anchor_coaxial_edges(tgt, tgt_key_map, tgt_edges)
        assert coax_pred
        assert coax_tgt
        coax_score = loops.loop_anchor_coaxial_score(
            pred, tgt, pred_key_map, tgt_key_map, pred_to_tgt, pred_edges, tgt_edges
        )
        assert coax_score == 1.0

        two_anchor = _FakeLoop(kind=LoopType.MULTILOOP, anchor_helices=[h1, h2], ordered_helices=[h1, h2])
        two_edges = loops._loop_anchor_coaxial_edges(two_anchor, pred_key_map, pred_edges)
        assert all(edge[0] in {0, 1} and edge[1] in {0, 1} for edge in two_edges)


class TestLoopConfusion:
    def test_loop_confusion_matrix_cases(self) -> None:
        device = torch.device("cpu")
        pred_loop = _FakeLoop(spans=[LoopSpan(1, 3)], pseudoknot_type=PseudoknotType.NONE, role="L1")
        tgt_loop = _FakeLoop(spans=[LoopSpan(2, 4)], pseudoknot_type=PseudoknotType.NONE, role="L1")
        mismatched = _FakeLoop(spans=[LoopSpan(2, 4)], pseudoknot_type=PseudoknotType.H_TYPE, role="L1")

        context_match = SimpleNamespace(device=device, loops_by_type={LoopType.HAIRPIN: ([pred_loop], [tgt_loop])})
        cm_match = loops._loop_confusion_matrix(LoopType.HAIRPIN, context_match)
        assert torch.allclose(cm_match, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))

        context_mismatch = SimpleNamespace(device=device, loops_by_type={LoopType.HAIRPIN: ([pred_loop], [mismatched])})
        cm_mismatch = loops._loop_confusion_matrix(LoopType.HAIRPIN, context_mismatch)
        assert torch.allclose(cm_mismatch, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

        context_empty = SimpleNamespace(device=device, loops_by_type={LoopType.HAIRPIN: ([], [])})
        cm_empty = loops._loop_confusion_matrix(LoopType.HAIRPIN, context_empty)
        assert bool(torch.isnan(cm_empty).all().item())

    def test_segment_confusion_matrix_filters_end_side(self) -> None:
        device = torch.device("cpu")
        pred_segments = [LoopSegment(LoopSegmentType.END, 0, 2, tier=0, side=EndSide.FIVE_PRIME)]
        tgt_segments_side_match = [LoopSegment(LoopSegmentType.END, 1, 3, tier=0, side=EndSide.FIVE_PRIME)]
        tgt_segments_side_mismatch = [LoopSegment(LoopSegmentType.END, 1, 3, tier=0, side=EndSide.THREE_PRIME)]
        tgt_segments_side_unknown = [LoopSegment(LoopSegmentType.END, 1, 3, tier=0, side=None)]
        pred_segments_side_unknown = [LoopSegment(LoopSegmentType.END, 0, 2, tier=0, side=None)]

        context_match = SimpleNamespace(
            device=device,
            loop_segments_by_kind={LoopSegmentType.END: (pred_segments, tgt_segments_side_match)},
        )
        cm_match = loops._segment_confusion_matrix(LoopSegmentType.END, context_match)
        assert torch.allclose(cm_match, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))

        context_mismatch = SimpleNamespace(
            device=device,
            loop_segments_by_kind={LoopSegmentType.END: (pred_segments, tgt_segments_side_mismatch)},
        )
        cm_mismatch = loops._segment_confusion_matrix(LoopSegmentType.END, context_mismatch)
        assert torch.allclose(cm_mismatch, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

        context_tgt_unknown = SimpleNamespace(
            device=device,
            loop_segments_by_kind={LoopSegmentType.END: (pred_segments, tgt_segments_side_unknown)},
        )
        cm_tgt_unknown = loops._segment_confusion_matrix(LoopSegmentType.END, context_tgt_unknown)
        assert torch.allclose(cm_tgt_unknown, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))

        context_pred_unknown = SimpleNamespace(
            device=device,
            loop_segments_by_kind={LoopSegmentType.END: (pred_segments_side_unknown, tgt_segments_side_mismatch)},
        )
        cm_pred_unknown = loops._segment_confusion_matrix(LoopSegmentType.END, context_pred_unknown)
        assert torch.allclose(cm_pred_unknown, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))

        context_empty = SimpleNamespace(device=device, loop_segments_by_kind={LoopSegmentType.END: ([], [])})
        cm_empty = loops._segment_confusion_matrix(LoopSegmentType.END, context_empty)
        assert bool(torch.isnan(cm_empty).all().item())


class TestLoopMetrics:
    def test_loop_and_segment_metrics_use_overlap_ratios(self, loop_overlap_context) -> None:
        loop_cm = loops.hairpin_loops_confusion(loop_overlap_context)
        confusion_loop_f1 = f1_from_confusion(loop_cm, loop_overlap_context.device)
        assert confusion_loop_f1 == pytest.approx(1.0)
        assert loops.hairpin_loops_precision(loop_overlap_context) == pytest.approx(0.5)
        assert loops.hairpin_loops_recall(loop_overlap_context) == pytest.approx(1.0)
        assert loops.hairpin_loops_f1(loop_overlap_context) == pytest.approx(2.0 / 3.0)
        assert loops.hairpin_loops_f1(loop_overlap_context) < confusion_loop_f1

        segment_cm = loops.hairpin_segments_confusion(loop_overlap_context)
        confusion_segment_f1 = f1_from_confusion(segment_cm, loop_overlap_context.device)
        assert confusion_segment_f1 == pytest.approx(1.0)
        assert loops.hairpin_segments_precision(loop_overlap_context) == pytest.approx(0.5)
        assert loops.hairpin_segments_recall(loop_overlap_context) == pytest.approx(1.0)
        assert loops.hairpin_segments_f1(loop_overlap_context) == pytest.approx(2.0 / 3.0)
        assert loops.hairpin_segments_f1(loop_overlap_context) < confusion_segment_f1

    @pytest.mark.parametrize(
        ("confusion", "precision", "recall", "f1"),
        [
            pytest.param(
                loops.bulge_nucleotides_confusion,
                loops.bulge_nucleotides_precision,
                loops.bulge_nucleotides_recall,
                loops.bulge_nucleotides_f1,
                id="bulge-nucleotides",
            ),
            pytest.param(
                loops.external_nucleotides_confusion,
                loops.external_nucleotides_precision,
                loops.external_nucleotides_recall,
                loops.external_nucleotides_f1,
                id="external-nucleotides",
            ),
            pytest.param(
                loops.hairpin_nucleotides_confusion,
                loops.hairpin_nucleotides_precision,
                loops.hairpin_nucleotides_recall,
                loops.hairpin_nucleotides_f1,
                id="hairpin-nucleotides",
            ),
            pytest.param(
                loops.internal_nucleotides_confusion,
                loops.internal_nucleotides_precision,
                loops.internal_nucleotides_recall,
                loops.internal_nucleotides_f1,
                id="internal-nucleotides",
            ),
            pytest.param(
                loops.multiloop_nucleotides_confusion,
                loops.multiloop_nucleotides_precision,
                loops.multiloop_nucleotides_recall,
                loops.multiloop_nucleotides_f1,
                id="multiloop-nucleotides",
            ),
        ],
    )
    def test_loop_nucleotide_metric_families(self, perfect_nested_context, confusion, precision, recall, f1) -> None:
        assert_confusion_metric_family(perfect_nested_context, confusion, precision, recall, f1)
