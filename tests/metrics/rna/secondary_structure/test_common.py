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

from types import SimpleNamespace

import pytest
import torch

from multimolecule.metrics.rna.secondary_structure import common
from multimolecule.utils.rna.secondary_structure import (
    EndSide,
    HelixSegment,
    LoopSegment,
    LoopSegmentType,
    LoopSpan,
    LoopType,
    PseudoknotType,
)


def test_anchor_pairs_score() -> None:
    assert common.anchor_pairs_score([], []) == 1.0
    assert common.anchor_pairs_score([(0, 5)], []) == 0.0
    assert common.anchor_pairs_score([(5, 0)], [(0, 5)]) == 1.0
    assert common.anchor_pairs_score([(0, 5)], [(1, 4)]) == 0.0


def test_pair_overlap_score_matrix_and_helix_score_matrix() -> None:
    pred_lengths = torch.tensor([2, 1], dtype=torch.long)
    pred_pair_ids = torch.tensor([10, 11, 20], dtype=torch.long)
    pred_owner = torch.tensor([0, 0, 1], dtype=torch.long)
    tgt_lengths = torch.tensor([2], dtype=torch.long)
    tgt_pair_ids = torch.tensor([11, 12], dtype=torch.long)
    tgt_owner = torch.tensor([0, 0], dtype=torch.long)

    score, inter = common.pair_overlap_score_matrix(
        pred_lengths,
        pred_pair_ids,
        pred_owner,
        tgt_lengths,
        tgt_pair_ids,
        tgt_owner,
        device=torch.device("cpu"),
    )
    assert score.shape == (2, 1)
    assert inter.shape == (2, 1)
    assert score[0, 0] == pytest.approx(1.0 / 3.0)
    assert float(inter[0, 0].item()) == 1.0
    assert float(score[1, 0].item()) == -1.0

    empty_score, empty_inter = common.pair_overlap_score_matrix(
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        tgt_lengths,
        tgt_pair_ids,
        tgt_owner,
        device=torch.device("cpu"),
    )
    assert empty_score.shape == (0, 1)
    assert empty_inter.shape == (0, 1)

    pred_helices = [HelixSegment(0, 1, 9, 8, ((0, 9), (1, 8)), 0)]
    tgt_helices = [HelixSegment(1, 2, 8, 7, ((1, 8), (2, 7)), 0)]
    helix_score = common.helix_score_matrix(pred_helices, tgt_helices, device=torch.device("cpu"))
    assert helix_score.shape == (1, 1)
    assert float(helix_score[0, 0].item()) > (1.0 / 3.0)

    helix_score_empty = common.helix_score_matrix([], tgt_helices, device=torch.device("cpu"))
    assert helix_score_empty.shape == (0, 1)


def test_loop_overlap_and_segment_type_mapping() -> None:
    pred_spans = [LoopSpan(0, 3), LoopSpan(6, 8)]
    tgt_spans = [LoopSpan(2, 5), LoopSpan(7, 9)]
    # overlap: (2,3)=2 and (7,8)=2
    assert common.loop_overlap_len(pred_spans, tgt_spans) == 4

    assert common.segment_type_from_loop(LoopType.HAIRPIN) == LoopSegmentType.HAIRPIN
    assert common.segment_type_from_loop(LoopType.BULGE) == LoopSegmentType.BULGE
    assert common.segment_type_from_loop(LoopType.INTERNAL) == LoopSegmentType.INTERNAL
    assert common.segment_type_from_loop(LoopType.MULTILOOP) == LoopSegmentType.BRANCH
    assert common.segment_type_from_loop(LoopType.EXTERNAL) == LoopSegmentType.EXTERNAL
    with pytest.raises(ValueError):
        common.segment_type_from_loop(None)  # type: ignore[arg-type]


def test_loop_taxonomy_and_segment_overlap_components() -> None:
    pred = SimpleNamespace(pseudoknot_type=PseudoknotType.H_TYPE, role="L1")
    tgt_unknown = SimpleNamespace(pseudoknot_type=PseudoknotType.UNKNOWN, role="L1")
    tgt_other = SimpleNamespace(pseudoknot_type=PseudoknotType.KISSING_HAIRPIN, role="L1")
    tgt_role = SimpleNamespace(pseudoknot_type=PseudoknotType.H_TYPE, role="L2")
    assert common.loop_taxonomy_compatible(pred, tgt_unknown)
    assert not common.loop_taxonomy_compatible(pred, tgt_other)
    assert not common.loop_taxonomy_compatible(pred, tgt_role)

    pred_segments = [
        LoopSegment(LoopSegmentType.END, 0, 2, tier=0, side=EndSide.FIVE_PRIME),
        LoopSegment(LoopSegmentType.END, 5, 6, tier=0, side=None),
    ]
    tgt_segments = [
        LoopSegment(LoopSegmentType.END, 1, 3, tier=0, side=EndSide.FIVE_PRIME),
        LoopSegment(LoopSegmentType.END, 5, 6, tier=0, side=EndSide.THREE_PRIME),
    ]
    pred_sizes, tgt_sizes, overlap, valid = common.segment_overlap_components(
        pred_segments,
        tgt_segments,
        device=torch.device("cpu"),
        enforce_end_side=True,
    )
    assert pred_sizes.tolist() == [3.0, 2.0]
    assert tgt_sizes.tolist() == [3.0, 2.0]
    assert overlap.tolist() == [[2.0, 0.0], [0.0, 2.0]]
    assert valid.tolist() == [[True, False], [False, True]]

    pred_sizes_empty, tgt_sizes_empty, overlap_empty, valid_empty = common.segment_overlap_components(
        [],
        tgt_segments,
        device=torch.device("cpu"),
        enforce_end_side=True,
    )
    assert pred_sizes_empty.shape == (0,)
    assert tgt_sizes_empty.shape == (2,)
    assert overlap_empty.shape == (0, 2)
    assert valid_empty.shape == (0, 2)


def test_confusion_exact_error_and_bipartite_matching() -> None:
    pred = torch.tensor([[0, 5], [1, 4]], dtype=torch.long)
    tgt = torch.tensor([[0, 5], [2, 3]], dtype=torch.long)

    cm = common.confusion_from_items(pred, tgt, torch.device("cpu"))
    assert torch.allclose(cm, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    assert common.pair_exact_match(pred, tgt).item() == pytest.approx(0.0)
    assert common.pair_error_rate(pred, tgt).item() == pytest.approx(2.0 / 3.0)

    empty = torch.empty((0, 2), dtype=torch.long)
    assert torch.isnan(common.pair_exact_match(empty, empty))
    assert torch.isnan(common.pair_error_rate(empty, empty))

    match_rows, match_cols = common.bipartite_match(torch.tensor([[0.9, -1.0], [0.1, 0.8]], dtype=torch.float32))
    assert match_rows.tolist() == [0, 1]
    assert match_cols.tolist() == [0, 1]

    no_rows, no_cols = common.bipartite_match(torch.full((2, 2), -1.0))
    assert no_rows.numel() == 0
    assert no_cols.numel() == 0


def test_pairs_pr_curve_and_graph_edit_distance() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.8, 0.2, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.7, 0.0],
        ],
        dtype=torch.float32,
    )
    target_pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    precision, recall, thresholds = common.pairs_precision_recall_curve(scores, target_pairs)
    assert precision.ndim == 1 and recall.ndim == 1 and thresholds.ndim == 1
    assert precision.numel() == thresholds.numel() + 1
    assert recall[-1] == pytest.approx(0.0)

    ged = common.approximate_graph_edit_distance(
        pred_nodes=["A"],
        tgt_nodes=["B"],
        pred_edges=[(0, 0, "x")],
        tgt_edges=[(0, 0, "y")],
        node_cost_fn=lambda a, b: 0.0 if a == b else 0.5,
        device=torch.device("cpu"),
    )
    assert float(ged.item()) > 0.0

    zero_ged = common.approximate_graph_edit_distance(
        pred_nodes=[],
        tgt_nodes=[],
        pred_edges=[],
        tgt_edges=[],
        node_cost_fn=lambda a, b: 0.0,
        device=torch.device("cpu"),
    )
    assert float(zero_ged.item()) == pytest.approx(0.0)


def test_confusion_scalar_reductions() -> None:
    cm = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    assert common.f1_from_confusion(cm, torch.device("cpu")).item() == pytest.approx(0.5)
    assert common.precision_from_confusion(cm, torch.device("cpu")).item() == pytest.approx(0.5)
    assert common.recall_from_confusion(cm, torch.device("cpu")).item() == pytest.approx(0.5)

    nan_cm = torch.zeros((2, 2), dtype=torch.float32)
    assert torch.isnan(common.f1_from_confusion(nan_cm, torch.device("cpu")))
    assert torch.isnan(common.precision_from_confusion(nan_cm, torch.device("cpu")))
    assert torch.isnan(common.recall_from_confusion(nan_cm, torch.device("cpu")))
