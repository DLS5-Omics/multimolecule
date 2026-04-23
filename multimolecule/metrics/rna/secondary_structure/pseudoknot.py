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

from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_confusion_matrix, binary_precision_recall_curve

from .common import (
    confusion_from_items,
    f1_from_confusion,
    pairs_precision_recall_curve,
    precision_from_confusion,
    recall_from_confusion,
)
from .functional import binary_f1_score, binary_mcc, binary_precision, binary_recall
from .stems import (
    event_stem_ids,
    stem_candidate_segments,
    stem_ids_from_arrays,
    stem_labels_from_gt,
    stem_match_indices,
    stems_from_arrays,
    subset_stem_data,
)

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def _map_event_ids(pred_event_ids: Tensor, matched_pred_ids: Tensor, matched_tgt_ids: Tensor) -> Tensor:
    if pred_event_ids.numel() == 0 or matched_pred_ids.numel() == 0:
        return pred_event_ids.new_empty((0, 2), dtype=torch.long)
    order = torch.argsort(matched_pred_ids)
    sorted_pred = matched_pred_ids[order]
    sorted_tgt = matched_tgt_ids[order]
    flat_ids = pred_event_ids.reshape(-1)
    idx = torch.searchsorted(sorted_pred, flat_ids)
    in_bounds = idx < sorted_pred.numel()
    valid = in_bounds.clone()
    if in_bounds.any():
        valid[in_bounds] = sorted_pred[idx[in_bounds]] == flat_ids[in_bounds]
    mapped_flat = flat_ids.new_full(flat_ids.shape, -1)
    mapped_flat[valid] = sorted_tgt[idx[valid]]
    return mapped_flat.view(-1, 2)


def crossing_events_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.crossing_events_confusion
    return f1_from_confusion(cm, context.device)


def crossing_events_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.crossing_events_confusion
    return precision_from_confusion(cm, context.device)


def crossing_events_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.crossing_events_confusion
    return recall_from_confusion(cm, context.device)


def crossing_events_precision_recall_curve(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor, Tensor]:
    pred = context.pred
    device = pred.device
    base = context.length + 1

    (
        cand_scores,
        cand_start_i,
        cand_start_j,
        cand_lengths,
        cand_pair_ids,
        cand_pair_stem,
    ) = stem_candidate_segments(pred)

    if cand_scores.numel() == 0:
        return binary_precision_recall_curve(
            torch.tensor([0.0], device=device, dtype=torch.float32),
            torch.tensor([0], device=device, dtype=torch.int64),
        )

    cand_stems = stems_from_arrays(cand_start_i, cand_start_j, cand_lengths)
    order = torch.argsort(cand_start_i * base + cand_start_j)
    cand_stems = cand_stems[order]
    cand_scores = cand_scores[order]

    num_stems = cand_stems.shape[0]
    tri = torch.triu_indices(num_stems, num_stems, offset=1, device=device)
    idx_i = tri[0]
    idx_j = tri[1]
    a5 = cand_stems[idx_i, 0]
    a3 = cand_stems[idx_i, 2]
    b5 = cand_stems[idx_j, 0]
    b3 = cand_stems[idx_j, 2]
    cross_mask = (a5 < b5) & (b5 < a3) & (a3 < b3)
    event_indices = torch.stack([idx_i[cross_mask], idx_j[cross_mask]], dim=1)

    if event_indices.numel() == 0:
        pred_events = cand_start_i.new_empty((0, 8), dtype=torch.long)
        pred_event_scores = cand_scores.new_empty((0,), dtype=torch.float32)
    else:
        ev_a = cand_stems[event_indices[:, 0]]
        ev_b = cand_stems[event_indices[:, 1]]
        events = torch.stack([ev_a, ev_b], dim=1)
        pred_events = events.reshape(-1, 8).to(dtype=torch.long)
        pred_event_scores = torch.minimum(cand_scores[event_indices[:, 0]], cand_scores[event_indices[:, 1]]).to(
            dtype=torch.float32
        )
        uniq_pred, inv_pred = torch.unique(pred_events, return_inverse=True, dim=0)
        if uniq_pred.shape[0] != pred_events.shape[0]:
            agg_scores = torch.zeros(uniq_pred.shape[0], dtype=torch.float32, device=device)
            agg_scores.scatter_reduce_(0, inv_pred, pred_event_scores, reduce="amax", include_self=False)
            pred_events = uniq_pred
            pred_event_scores = agg_scores

    gt_events = context.target_crossing_events

    if pred_events.numel() == 0:
        return binary_precision_recall_curve(
            torch.tensor([0.0], device=device, dtype=torch.float32),
            torch.tensor([0], device=device, dtype=torch.int64),
        )
    if gt_events.numel() == 0:
        labels_t = torch.zeros_like(pred_event_scores, dtype=torch.int64)
        return binary_precision_recall_curve(pred_event_scores, labels_t)

    tgt_mask = torch.isin(context.target_stem_ids, context.target_crossing_stem_ids)
    tgt_start_i = context.target_stem_start_i[tgt_mask]
    tgt_start_j = context.target_stem_start_j[tgt_mask]
    tgt_lengths = context.target_stem_lengths[tgt_mask]
    if tgt_lengths.numel() == 0:
        labels_t = torch.zeros_like(pred_event_scores, dtype=torch.int64)
        return binary_precision_recall_curve(pred_event_scores, labels_t)

    tgt_lengths_sub, tgt_pair_ids, tgt_pair_stem = subset_stem_data(
        context.target_stem_lengths,
        context.target_stem_pair_ids,
        context.target_stem_pair_stem,
        tgt_mask,
    )
    matched_pred, matched_tgt, _ = stem_match_indices(
        cand_lengths,
        cand_pair_ids,
        cand_pair_stem,
        tgt_lengths_sub,
        tgt_pair_ids,
        tgt_pair_stem,
    )
    cand_ids = stem_ids_from_arrays(cand_start_i, cand_start_j, cand_lengths, base)
    tgt_ids = stem_ids_from_arrays(tgt_start_i, tgt_start_j, tgt_lengths, base)
    matched_pred_ids = cand_ids[matched_pred]
    matched_tgt_ids = tgt_ids[matched_tgt]

    target_event_ids = event_stem_ids(gt_events, base)
    if target_event_ids.numel():
        target_event_ids = torch.unique(target_event_ids, dim=0)
    pred_event_ids = event_stem_ids(pred_events, base)
    labels = torch.zeros(pred_event_ids.shape[0], device=device, dtype=torch.int64)
    mapped = _map_event_ids(pred_event_ids, matched_pred_ids, matched_tgt_ids)
    mapped_mask = (mapped >= 0).all(dim=1)
    if mapped_mask.any():
        mapped_valid = mapped[mapped_mask]
        mapped_sorted = torch.sort(mapped_valid, dim=1).values
        target_sorted = torch.sort(target_event_ids, dim=1).values
        matches = (mapped_sorted[:, None, :] == target_sorted[None, :, :]).all(dim=2)
        labels[mapped_mask] = matches.any(dim=1).to(dtype=torch.int64)
    return binary_precision_recall_curve(pred_event_scores, labels)


def crossing_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = context.crossing_nt_labels
    return binary_confusion_matrix(preds, targets)


def crossing_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = context.crossing_nt_labels
    return binary_f1_score(preds, targets)


def crossing_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = context.crossing_nt_labels
    return binary_mcc(preds, targets)


def crossing_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = context.crossing_nt_labels
    return binary_precision(preds, targets)


def crossing_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = context.crossing_nt_labels
    return binary_recall(preds, targets)


def crossing_nucleotides_precision_recall_curve(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor, Tensor]:
    pred = context.pred
    device = pred.device
    scores_nt = _crossing_nt_scores(pred)
    if scores_nt.numel() == 0:
        return binary_precision_recall_curve(
            torch.tensor([0.0], device=device, dtype=torch.float32),
            torch.tensor([0], device=device, dtype=torch.int64),
        )

    length = context.length
    target_idx = context.target_topology.crossing_nucleotides
    y_nt = torch.zeros(length, dtype=torch.int64, device=device)
    if target_idx.numel():
        y_nt[target_idx] = 1
    return binary_precision_recall_curve(scores_nt, y_nt)


def _crossing_nt_scores(pred: Tensor) -> Tensor:
    if pred.ndim != 2 or pred.shape[0] != pred.shape[1]:
        raise ValueError("pred must be a square 2D contact map")
    n = int(pred.shape[0])
    if n == 0:
        return pred.new_empty((0,), dtype=torch.float32)

    scores = pred.to(dtype=torch.float32)
    scores = (scores + scores.T) / 2
    scores = scores.clone()
    scores.fill_diagonal_(0)

    best = torch.zeros(n, device=scores.device, dtype=torch.float32)
    if n <= 2:
        return best

    neg_inf = scores.new_full((), float("-inf"))
    row_idx = torch.arange(n, device=scores.device)[:, None]
    col_idx = torch.arange(n, device=scores.device)[None, :]
    upper_offset2 = col_idx >= (row_idx + 2)

    # Right component: max_{i<r<j, c>j} scores[r, c]
    # suffix[r, c] = max_{c' > c} scores[r, c']
    suffix = scores.new_full((n, n), float("-inf"))
    suffix[:, :-1] = torch.cummax(scores[:, 1:].flip(1), dim=1).values.flip(1)
    # Restrict to r < j (crossing condition): zero out entries where row >= col
    restricted = suffix.masked_fill(row_idx >= col_idx, neg_inf)
    # Reverse cummax along rows: downward[r, j] = max_{r' >= r, r' < j} suffix[r', j]
    downward = torch.cummax(restricted.flip(0), dim=0).values.flip(0)
    # Shift: partner_right[i, j] = max_{i < r < j, c > j} scores[r, c]
    partner_right = scores.new_full((n, n), float("-inf"))
    partner_right[:-1, :] = downward[1:, :]
    cand_right = torch.minimum(scores, partner_right).masked_fill(~upper_offset2, neg_inf)
    row_max_right = cand_right.max(dim=1).values
    col_max_right = cand_right.max(dim=0).values
    best = torch.maximum(
        best,
        torch.where(torch.isfinite(row_max_right), row_max_right, torch.zeros_like(row_max_right)),
    )
    best = torch.maximum(
        best,
        torch.where(torch.isfinite(col_max_right), col_max_right, torch.zeros_like(col_max_right)),
    )

    # Left component: max_{r<i, i<c<j} scores[r, c]
    above = torch.cummax(scores, dim=0).values
    prior_rows = scores.new_full((n, n), float("-inf"))
    prior_rows[1:, :] = above[:-1, :]
    prior_rows = prior_rows.masked_fill(col_idx <= row_idx, neg_inf)
    prior_prefix = torch.cummax(prior_rows, dim=1).values
    partner_left = scores.new_full((n, n), float("-inf"))
    partner_left[:, 1:] = prior_prefix[:, :-1]
    valid_left = upper_offset2 & (row_idx >= 1)
    cand_left = torch.minimum(scores, partner_left).masked_fill(~valid_left, neg_inf)
    row_max_left = cand_left.max(dim=1).values
    col_max_left = cand_left.max(dim=0).values
    best = torch.maximum(
        best,
        torch.where(torch.isfinite(row_max_left), row_max_left, torch.zeros_like(row_max_left)),
    )
    best = torch.maximum(
        best,
        torch.where(torch.isfinite(col_max_left), col_max_left, torch.zeros_like(col_max_left)),
    )
    return best


def crossing_pairs_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    pred_crossing = context.pred_topology.crossing_pairs
    target_crossing = context.target_topology.crossing_pairs
    return confusion_from_items(pred_crossing, target_crossing, pred_crossing.device)


def crossing_pairs_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = crossing_pairs_confusion(context)
    return f1_from_confusion(cm, context.device)


def crossing_pairs_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = crossing_pairs_confusion(context)
    return precision_from_confusion(cm, context.device)


def crossing_pairs_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = crossing_pairs_confusion(context)
    return recall_from_confusion(cm, context.device)


def crossing_pairs_precision_recall_curve(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor, Tensor]:
    return pairs_precision_recall_curve(context.pred, context.target_topology.crossing_pairs)


def crossing_stem_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.crossing_stem_confusion
    return f1_from_confusion(cm, context.device)


def crossing_stem_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.crossing_stem_confusion
    return precision_from_confusion(cm, context.device)


def crossing_stem_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.crossing_stem_confusion
    return recall_from_confusion(cm, context.device)


def crossing_stem_precision_recall_curve(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor, Tensor]:
    pred = context.pred
    device = pred.device

    (
        candidate_scores,
        _,
        _,
        cand_lengths,
        cand_pair_ids,
        cand_pair_stem,
    ) = stem_candidate_segments(pred)
    if candidate_scores.numel() == 0:
        return binary_precision_recall_curve(
            torch.tensor([0.0], device=device, dtype=torch.float32),
            torch.tensor([0], device=device, dtype=torch.int64),
        )

    gt_lengths = context.target_stem_lengths
    gt_mask = torch.isin(context.target_stem_ids, context.target_crossing_stem_ids)

    labels = stem_labels_from_gt(
        cand_lengths,
        gt_lengths,
        cand_pair_ids=cand_pair_ids,
        cand_pair_stem=cand_pair_stem,
        gt_pair_ids=context.target_stem_pair_ids,
        gt_pair_stem=context.target_stem_pair_stem,
        gt_mask=gt_mask,
    )

    scores_t = candidate_scores.to(dtype=torch.float32)
    labels_t = labels.to(dtype=torch.int64)
    return binary_precision_recall_curve(scores_t, labels_t)
