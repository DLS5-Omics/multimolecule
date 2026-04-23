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

import pytest
import torch

from multimolecule.metrics.rna.secondary_structure import stems
from multimolecule.utils.rna.secondary_structure import StemSegment
from tests.metrics.rna.secondary_structure.utils import (
    assert_confusion_metric_family,
    assert_precision_recall_curve_contract,
    context_from_dot_bracket,
)


def test_stem_ids_are_stable_for_stems_and_events() -> None:
    stems_tensor = torch.tensor([[0, 1, 9, 8], [2, 3, 7, 6]], dtype=torch.long)
    ids = stems._stem_ids_from_stems(stems_tensor, base=20)
    assert ids.shape == (2,)
    assert ids[0] != ids[1]

    empty_ids = stems._stem_ids_from_stems(torch.empty((0, 4), dtype=torch.long), base=20)
    assert empty_ids.numel() == 0

    events = torch.tensor(
        [
            [[0, 1, 9, 8], [2, 3, 7, 6]],
            [[2, 3, 7, 6], [4, 5, 5, 4]],
        ],
        dtype=torch.long,
    )
    event_pair_ids = stems.event_stem_ids(events, base=20)
    assert event_pair_ids.shape == (2, 2)

    unique_ids = stems.stem_ids_from_events(events, base=20)
    assert unique_ids.numel() == 3


def test_stem_segment_data_expands_pairs_and_segments() -> None:
    pairs = torch.tensor([[0, 5], [1, 4], [2, 3]], dtype=torch.long)
    start_i, start_j, lengths, pair_ids, pair_stem = stems.stem_segment_data(pairs, base=10)
    assert start_i.numel() == 1
    assert start_j.numel() == 1
    assert lengths.tolist() == [3]
    assert pair_ids.numel() == 3
    assert pair_stem.tolist() == [0, 0, 0]

    empty = torch.empty((0, 2), dtype=torch.long)
    out = stems.stem_segment_data(empty, base=10)
    assert all(item.numel() == 0 for item in out)

    segs = [StemSegment(0, 1, 9, 8, tier=0), StemSegment(3, 4, 6, 5, tier=0)]
    lengths_seg, pair_ids_seg, owners_seg = stems.stem_pair_data_from_segments(
        segs, base=20, device=torch.device("cpu")
    )
    assert lengths_seg.tolist() == [2, 2]
    assert pair_ids_seg.numel() == 4
    assert owners_seg.tolist() == [0, 0, 1, 1]

    empty_lengths, empty_pair_ids, empty_owners = stems.stem_pair_data_from_segments(
        [], base=20, device=torch.device("cpu")
    )
    assert empty_lengths.numel() == 0
    assert empty_pair_ids.numel() == 0
    assert empty_owners.numel() == 0


def test_stem_edge_adjacency_and_subset_data() -> None:
    class _Edge:
        def __init__(self, src: int, dst: int, edge_type: int) -> None:
            self.src = src
            self.dst = dst
            self.type = edge_type

    edge_list = [
        _Edge(0, 1, 1),
        (1, 0, 2),
        (-1, 0, 1),  # out-of-range
        (0, 3, 1),  # out-of-range
        (0, 1, 999),  # unknown type
    ]
    adj = stems.stem_edge_adjacency_by_type(edge_list, stem_count=2)
    assert adj[0][1] == {1}
    assert adj[1][2] == {0}

    lengths = torch.tensor([2, 3], dtype=torch.long)
    pair_ids = torch.tensor([11, 12, 13, 21, 22], dtype=torch.long)
    pair_owner = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    mask = torch.tensor([False, True], dtype=torch.bool)
    new_lengths, new_pair_ids, new_owner = stems.subset_stem_data(lengths, pair_ids, pair_owner, mask)
    assert new_lengths.tolist() == [3]
    assert new_pair_ids.tolist() == [21, 22]
    assert new_owner.tolist() == [0, 0]

    empty_lengths, empty_pair_ids, empty_owner = stems.subset_stem_data(
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.bool),
    )
    assert empty_lengths.numel() == 0
    assert empty_pair_ids.numel() == 0
    assert empty_owner.numel() == 0


def test_stem_match_indices_and_confusion_from_arrays() -> None:
    pred_lengths = torch.tensor([2, 1], dtype=torch.long)
    pred_pair_ids = torch.tensor([11, 12, 20], dtype=torch.long)
    pred_pair_stem = torch.tensor([0, 0, 1], dtype=torch.long)
    tgt_lengths = torch.tensor([2], dtype=torch.long)
    tgt_pair_ids = torch.tensor([12, 13], dtype=torch.long)
    tgt_pair_stem = torch.tensor([0, 0], dtype=torch.long)

    matched_pred, matched_tgt, inter = stems.stem_match_indices(
        pred_lengths,
        pred_pair_ids,
        pred_pair_stem,
        tgt_lengths,
        tgt_pair_ids,
        tgt_pair_stem,
    )
    assert matched_pred.tolist() == [0]
    assert matched_tgt.tolist() == [0]
    assert inter.tolist() == [1.0]

    matched_pred_unweighted, _, _ = stems.stem_match_indices(
        pred_lengths,
        pred_pair_ids,
        pred_pair_stem,
        tgt_lengths,
        tgt_pair_ids,
        tgt_pair_stem,
        weight_by_length=False,
    )
    assert matched_pred_unweighted.tolist() == [0]

    empty_idx, empty_tgt, empty_inter = stems.stem_match_indices(
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        tgt_lengths,
        tgt_pair_ids,
        tgt_pair_stem,
    )
    assert empty_idx.numel() == 0
    assert empty_tgt.numel() == 0
    assert empty_inter.numel() == 0

    cm = stems.stem_confusion_from_arrays(
        pred_lengths,
        pred_pair_ids,
        pred_pair_stem,
        tgt_lengths,
        tgt_pair_ids,
        tgt_pair_stem,
    )
    assert torch.allclose(cm, torch.tensor([[0.0, 2.0], [1.0, 1.0]]))

    cm_empty = stems.stem_confusion_from_arrays(
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
    )
    assert torch.allclose(cm_empty, torch.zeros((2, 2)))


def test_stem_candidate_segments_and_labels_from_gt() -> None:
    n = 7
    scores = torch.full((n, n), 0.01, dtype=torch.float32)
    scores.fill_diagonal_(0.0)
    # Greedy decoding keeps high-confidence pairs; stem segmentation allows bulges.
    scores[0, 6] = scores[6, 0] = 0.99
    scores[1, 5] = scores[5, 1] = 0.98
    scores[3, 4] = scores[4, 3] = 0.97

    with pytest.warns(UserWarning):
        cand_scores, start_i, start_j, lengths, cand_pair_ids, cand_pair_stem = stems.stem_candidate_segments(scores)
    assert cand_scores.numel() == 1
    assert int(lengths[0].item()) >= 2
    assert start_i.tolist() == [0]
    assert start_j.tolist() == [6]
    assert 0.0 <= float(cand_scores[0].item()) <= 1.0
    assert cand_pair_ids.numel() == int(lengths[0].item())
    assert cand_pair_stem.tolist() == [0] * int(lengths[0].item())

    # With n=3 at most one pair is decoded, so all candidates are filtered by min stem length (>=2).
    scores_short = torch.full((3, 3), 0.2, dtype=torch.float32)
    scores_short.fill_diagonal_(0.0)
    with pytest.warns(UserWarning):
        empty_out = stems.stem_candidate_segments(scores_short)
    assert all(item.numel() == 0 for item in empty_out)

    labels = stems.stem_labels_from_gt(
        lengths,
        lengths.clone(),
        cand_pair_ids=cand_pair_ids,
        cand_pair_stem=cand_pair_stem,
        gt_pair_ids=cand_pair_ids.clone(),
        gt_pair_stem=cand_pair_stem.clone(),
    )
    assert labels.tolist() == [1]

    labels_masked = stems.stem_labels_from_gt(
        lengths,
        lengths.clone(),
        cand_pair_ids=cand_pair_ids,
        cand_pair_stem=cand_pair_stem,
        gt_pair_ids=cand_pair_ids.clone(),
        gt_pair_stem=cand_pair_stem.clone(),
        gt_mask=torch.tensor([False], dtype=torch.bool),
    )
    assert labels_masked.tolist() == [0]

    empty_labels = stems.stem_labels_from_gt(
        torch.empty((0,), dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        cand_pair_ids=torch.empty((0,), dtype=torch.long),
        cand_pair_stem=torch.empty((0,), dtype=torch.long),
        gt_pair_ids=torch.tensor([1], dtype=torch.long),
        gt_pair_stem=torch.tensor([0], dtype=torch.long),
    )
    assert empty_labels.numel() == 0


def test_stem_metric_families_use_confusion_matrices(perfect_nested_context) -> None:
    assert_confusion_metric_family(
        perfect_nested_context,
        stems.stem_confusion,
        stems.stem_precision,
        stems.stem_recall,
        stems.stem_f1,
    )
    assert_confusion_metric_family(
        perfect_nested_context,
        stems.stem_pairs_confusion,
        stems.stem_pairs_precision,
        stems.stem_pairs_recall,
        stems.stem_pairs_f1,
    )


def test_stem_confusion_is_cached_on_context(perfect_nested_context) -> None:
    first = perfect_nested_context.stem_confusion
    second = perfect_nested_context.stem_confusion
    assert first is second
    assert stems.stem_pairs_confusion(perfect_nested_context) is first


@pytest.mark.parametrize(
    ("dot_bracket", "sequence"),
    [
        pytest.param("(([[))]]", "ACGAUGUC", id="positive"),
        pytest.param("....", "AAAA", id="empty"),
    ],
)
def test_stem_precision_recall_curve_contract(dot_bracket: str, sequence: str) -> None:
    context = context_from_dot_bracket(dot_bracket, sequence=sequence)
    assert_precision_recall_curve_contract(stems.stem_precision_recall_curve(context))
