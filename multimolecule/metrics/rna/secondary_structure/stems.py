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

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_precision_recall_curve

from ....utils.rna.secondary_structure import StemEdgeType, contact_map_to_pairs, pairs_to_stem_segment_arrays
from .common import (
    MATCH_TIEBREAK_OVERLAP,
    MATCH_TIEBREAK_SIZE,
    bipartite_match,
    f1_from_confusion,
    make_confusion_matrix,
    pack_neighbors,
    pair_overlap_score_matrix,
    precision_from_confusion,
    recall_from_confusion,
)

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext

# Fixed evaluation constants; keep internal to avoid tunable metrics.
_STEM_PR_MIN_LEN = 2
_STEM_PR_SCORE_AGG = "min"


def stems_from_arrays(start_i: Tensor, start_j: Tensor, lengths: Tensor) -> Tensor:
    stop_i = start_i + lengths - 1
    stop_j = start_j - lengths + 1
    return torch.stack([start_i, stop_i, start_j, stop_j], dim=1)


def stem_ids_from_arrays(start_i: Tensor, start_j: Tensor, lengths: Tensor, base: int) -> Tensor:
    stop_i = start_i + lengths - 1
    stop_j = start_j - lengths + 1
    base = int(base)
    return (((start_i * base + stop_i) * base + start_j) * base) + stop_j


def _stem_ids_from_stems(stems: Tensor, base: int) -> Tensor:
    if stems.numel() == 0:
        return stems.new_empty((0,), dtype=torch.long)
    stems = stems.to(dtype=torch.long)
    base = int(base)
    return (((stems[:, 0] * base + stems[:, 1]) * base + stems[:, 2]) * base) + stems[:, 3]


def event_stem_ids(events: Tensor, base: int) -> Tensor:
    if events.numel() == 0:
        return events.new_empty((0, 2), dtype=torch.long)
    stems = events.reshape(-1, 2, 4).to(dtype=torch.long)
    stem_a = _stem_ids_from_stems(stems[:, 0], base)
    stem_b = _stem_ids_from_stems(stems[:, 1], base)
    return torch.stack([stem_a, stem_b], dim=1)


def stem_ids_from_events(events: Tensor, base: int) -> Tensor:
    if events.numel() == 0:
        return events.new_empty((0,), dtype=torch.long)
    base = int(base)
    stems = events.to(dtype=torch.long)
    a = stems[:, 0]
    b = stems[:, 1]
    stem_a = (((a[:, 0] * base + a[:, 1]) * base + a[:, 2]) * base) + a[:, 3]
    stem_b = (((b[:, 0] * base + b[:, 1]) * base + b[:, 2]) * base) + b[:, 3]
    return torch.unique(torch.cat([stem_a, stem_b]))


def stem_segment_data(pairs: Tensor, base: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if pairs.numel() == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty, empty
    start_i, start_j, lengths = pairs_to_stem_segment_arrays(pairs)
    if start_i.numel() == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty, empty
    num_stems = int(start_i.numel())
    total_pairs = lengths.sum()
    stem_ids = torch.repeat_interleave(torch.arange(num_stems, device=pairs.device), lengths)
    prefix = lengths.cumsum(0)
    offsets = torch.arange(total_pairs, device=pairs.device) - torch.repeat_interleave(prefix - lengths, lengths)
    pair_i = start_i[stem_ids] + offsets
    pair_j = start_j[stem_ids] - offsets
    base = int(base)
    pair_ids = (pair_i * base + pair_j).to(dtype=torch.long)
    return start_i, start_j, lengths, pair_ids, stem_ids.to(dtype=torch.long)


def stem_pair_data_from_segments(segments, base: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    if not segments:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty
    starts_5p = torch.tensor([seg.start_5p for seg in segments], device=device, dtype=torch.long)
    starts_3p = torch.tensor([seg.start_3p for seg in segments], device=device, dtype=torch.long)
    lengths = torch.tensor([len(seg) for seg in segments], device=device, dtype=torch.long)
    total_pairs = lengths.sum()
    if total_pairs.numel() == 0 or int(total_pairs.item()) == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return lengths, empty, empty
    stem_ids = torch.repeat_interleave(torch.arange(len(segments), device=device), lengths)
    offsets = torch.arange(int(total_pairs.item()), device=device) - torch.repeat_interleave(
        lengths.cumsum(0) - lengths, lengths
    )
    pair_i = starts_5p[stem_ids] + offsets
    pair_j = starts_3p[stem_ids] - offsets
    pair_ids = (pair_i * base + pair_j).to(dtype=torch.long)
    return lengths, pair_ids, stem_ids.to(dtype=torch.long)


def stem_edge_adjacency_by_type(edges, stem_count: int) -> dict[int, dict[int, set[int]]]:
    adj: dict[int, dict[int, set[int]]] = {
        idx: {int(edge_type): set() for edge_type in StemEdgeType} for idx in range(stem_count)
    }
    for edge in edges:
        if hasattr(edge, "src"):
            src = int(edge.src)
            dst = int(edge.dst)
            edge_type = int(edge.type)
        else:
            src, dst, edge_type = edge
        if src < 0 or dst < 0 or src >= stem_count or dst >= stem_count:
            continue
        if edge_type not in adj[src]:
            continue
        adj[src][edge_type].add(dst)
    return adj


def _stem_adjacency_score_matrix(
    pred_adj: dict[int, dict[int, set[int]]],
    tgt_adj: dict[int, dict[int, set[int]]],
    base_score: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    pred_count, tgt_count = base_score.shape
    if pred_count == 0 or tgt_count == 0:
        return torch.empty((pred_count, tgt_count), device=device, dtype=torch.float32)
    edge_types = [int(edge_type) for edge_type in StemEdgeType]
    if not edge_types:
        return torch.zeros((pred_count, tgt_count), device=device, dtype=torch.float32)
    valid_pairs = base_score >= 0
    if not bool(valid_pairs.any().item()):
        return torch.zeros((pred_count, tgt_count), device=device, dtype=torch.float32)

    scores = base_score
    if scores.device != device:
        scores = scores.to(device=device)
    if scores.dtype != torch.float32:
        scores = scores.to(dtype=torch.float32)
    score_mat = torch.zeros((pred_count, tgt_count), dtype=torch.float32, device=device)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    one = torch.tensor(1.0, device=device, dtype=torch.float32)

    for edge_type in edge_types:
        pred_neighbors = [sorted(pred_adj.get(i, {}).get(edge_type, set())) for i in range(pred_count)]
        tgt_neighbors = [sorted(tgt_adj.get(j, {}).get(edge_type, set())) for j in range(tgt_count)]
        pred_lengths, pred_padded = pack_neighbors(pred_neighbors, device)
        tgt_lengths, tgt_padded = pack_neighbors(tgt_neighbors, device)
        pred_len = pred_lengths[:, None]
        tgt_len = tgt_lengths[None, :]
        typed = torch.zeros((pred_count, tgt_count), dtype=torch.float32, device=device)
        typed[(pred_len == 0) & (tgt_len == 0)] = 1.0

        mask_p1 = (pred_len == 1) & (tgt_len >= 1)
        if mask_p1.any():
            pair_idx = torch.nonzero(mask_p1, as_tuple=False)
            pi = pair_idx[:, 0]
            tj = pair_idx[:, 1]
            pred_ids = pred_padded[pi, 0]
            tgt_ids = tgt_padded[tj]
            valid = tgt_ids >= 0
            gather_tgt = torch.where(valid, tgt_ids, torch.zeros_like(tgt_ids))
            vals = scores[pred_ids[:, None], gather_tgt]
            vals = vals.masked_fill(~valid, neg_inf)
            max_val = vals.max(dim=1).values
            denom = tgt_lengths[tj].to(dtype=torch.float32)
            typed[pi, tj] = torch.where(max_val >= 0, torch.minimum(max_val / denom, one), typed[pi, tj])

        mask_t1 = (pred_len > 1) & (tgt_len == 1)
        if mask_t1.any():
            pair_idx = torch.nonzero(mask_t1, as_tuple=False)
            pi = pair_idx[:, 0]
            tj = pair_idx[:, 1]
            pred_ids = pred_padded[pi]
            valid = pred_ids >= 0
            gather_pred = torch.where(valid, pred_ids, torch.zeros_like(pred_ids))
            vals = scores[gather_pred, tgt_padded[tj, 0][:, None]]
            vals = vals.masked_fill(~valid, neg_inf)
            max_val = vals.max(dim=1).values
            denom = pred_lengths[pi].to(dtype=torch.float32)
            typed[pi, tj] = torch.where(max_val >= 0, torch.minimum(max_val / denom, one), typed[pi, tj])

        mask_general = (pred_len > 1) & (tgt_len > 1)
        if mask_general.any():
            pair_idx = torch.nonzero(mask_general, as_tuple=False)
            for pair in pair_idx:
                i = int(pair[0].item())
                j = int(pair[1].item())
                pred_ids = pred_padded[i]
                tgt_ids = tgt_padded[j]
                pred_ids = pred_ids[pred_ids >= 0]
                tgt_ids = tgt_ids[tgt_ids >= 0]
                if pred_ids.numel() == 0 or tgt_ids.numel() == 0:
                    continue
                sub = scores[pred_ids][:, tgt_ids]
                if not bool((sub >= 0).any().item()):
                    continue
                matched_pred, matched_tgt = bipartite_match(sub)
                if matched_pred.numel() == 0:
                    continue
                sim_sum = sub[matched_pred, matched_tgt].sum()
                denom = float(max(pred_ids.numel(), tgt_ids.numel()))
                if denom > 0:
                    typed[i, j] = torch.minimum(sim_sum / denom, one)

        score_mat += typed

    score_mat = score_mat / float(len(edge_types))
    if valid_pairs.device != device:
        valid_pairs = valid_pairs.to(device=device)
    score_mat = torch.where(valid_pairs, score_mat, torch.zeros_like(score_mat))
    return score_mat


def stem_score_matrix(
    pred_lengths: Tensor,
    pred_pair_ids: Tensor,
    pred_pair_stem: Tensor,
    tgt_lengths: Tensor,
    tgt_pair_ids: Tensor,
    tgt_pair_stem: Tensor,
    pred_adj: dict[int, dict[int, set[int]]],
    tgt_adj: dict[int, dict[int, set[int]]],
    *,
    device: torch.device,
) -> Tensor:
    base_score, inter_mat = pair_overlap_score_matrix(
        pred_lengths,
        pred_pair_ids,
        pred_pair_stem,
        tgt_lengths,
        tgt_pair_ids,
        tgt_pair_stem,
        device=device,
    )
    if base_score.numel() == 0:
        return base_score
    adj_score = _stem_adjacency_score_matrix(pred_adj, tgt_adj, base_score, device=device)
    score_mat = torch.full_like(base_score, -1.0, dtype=torch.float32)
    valid = base_score >= 0
    if valid.any():
        score_mat[valid] = (base_score[valid] + adj_score[valid]) / 2.0
        score_mat[valid] += inter_mat[valid] * MATCH_TIEBREAK_OVERLAP
        tgt_size_mat = tgt_lengths.unsqueeze(0).to(dtype=torch.float32).expand_as(score_mat)
        score_mat[valid] += tgt_size_mat[valid] * MATCH_TIEBREAK_SIZE
    return score_mat


def _stem_topology_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_segments, pred_edges = context.pred_topology.stem_graph_components()
    tgt_segments, tgt_edges = context.target_topology.stem_graph_components()
    pred_total = len(pred_segments)
    tgt_total = len(tgt_segments)

    if pred_total == 0 and tgt_total == 0:
        return torch.zeros((2, 2), dtype=torch.float32, device=device)
    if pred_total == 0:
        tgt_total_t = torch.tensor(float(tgt_total), device=device)
        zero = tgt_total_t.new_zeros(())
        return make_confusion_matrix(zero, zero, tgt_total_t)
    if tgt_total == 0:
        pred_total_t = torch.tensor(float(pred_total), device=device)
        zero = pred_total_t.new_zeros(())
        return make_confusion_matrix(zero, pred_total_t, zero)

    base = context.length + 1
    matched = 0
    if pred_segments and tgt_segments:
        pred_lengths, pred_pair_ids, pred_pair_stem = stem_pair_data_from_segments(pred_segments, base, device)
        tgt_lengths, tgt_pair_ids, tgt_pair_stem = stem_pair_data_from_segments(tgt_segments, base, device)
        pred_adj = stem_edge_adjacency_by_type(pred_edges, len(pred_segments))
        tgt_adj = stem_edge_adjacency_by_type(tgt_edges, len(tgt_segments))
        score_mat = stem_score_matrix(
            pred_lengths,
            pred_pair_ids,
            pred_pair_stem,
            tgt_lengths,
            tgt_pair_ids,
            tgt_pair_stem,
            pred_adj,
            tgt_adj,
            device=device,
        )
        matched_pred, _ = bipartite_match(score_mat)
        matched = int(matched_pred.numel())

    tp = torch.tensor(float(matched), device=device)
    pred_total_t = torch.tensor(float(pred_total), device=device)
    tgt_total_t = torch.tensor(float(tgt_total), device=device)
    fp = pred_total_t - tp
    fn = tgt_total_t - tp
    return make_confusion_matrix(tp, fp, fn)


def subset_stem_data(
    lengths: Tensor, pair_ids: Tensor, pair_stem: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    if lengths.numel() == 0:
        empty = lengths.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if mask.numel() == 0 or idx.numel() == 0:
        empty = lengths.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    new_lengths = lengths[mask]
    remap = lengths.new_full((lengths.shape[0],), -1, dtype=torch.long)
    remap[idx] = torch.arange(idx.numel(), device=lengths.device)
    mapped = remap[pair_stem]
    pair_mask = mapped >= 0
    return new_lengths, pair_ids[pair_mask], mapped[pair_mask]


def _stem_arrays_crossing(
    context: RnaSecondaryStructureContext,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pred_mask = torch.isin(context.pred_stem_ids, context.pred_crossing_stem_ids)
    tgt_mask = torch.isin(context.target_stem_ids, context.target_crossing_stem_ids)

    pred_lengths, pred_pair_ids, pred_pair_stem = subset_stem_data(
        context.pred_stem_lengths,
        context.pred_stem_pair_ids,
        context.pred_stem_pair_stem,
        pred_mask,
    )
    tgt_lengths, tgt_pair_ids, tgt_pair_stem = subset_stem_data(
        context.target_stem_lengths,
        context.target_stem_pair_ids,
        context.target_stem_pair_stem,
        tgt_mask,
    )
    return pred_lengths, pred_pair_ids, pred_pair_stem, tgt_lengths, tgt_pair_ids, tgt_pair_stem


def stem_match_indices(
    pred_lengths: Tensor,
    pred_pair_ids: Tensor,
    pred_pair_stem: Tensor,
    tgt_lengths: Tensor,
    tgt_pair_ids: Tensor,
    tgt_pair_stem: Tensor,
    *,
    weight_by_length: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    if pred_lengths.numel() == 0 or tgt_lengths.numel() == 0:
        empty_idx = pred_lengths.new_empty((0,), dtype=torch.long)
        empty_inter = pred_lengths.new_empty((0,), dtype=torch.float32)
        return empty_idx, empty_idx, empty_inter
    if pred_pair_ids.numel() == 0 or tgt_pair_ids.numel() == 0:
        empty_idx = pred_lengths.new_empty((0,), dtype=torch.long)
        empty_inter = pred_lengths.new_empty((0,), dtype=torch.float32)
        return empty_idx, empty_idx, empty_inter

    all_ids = torch.cat([pred_pair_ids, tgt_pair_ids], dim=0)
    uniq_ids, counts = torch.unique(all_ids, return_counts=True)
    common_ids = uniq_ids[counts > 1]
    if common_ids.numel() == 0:
        empty_idx = pred_lengths.new_empty((0,), dtype=torch.long)
        empty_inter = pred_lengths.new_empty((0,), dtype=torch.float32)
        return empty_idx, empty_idx, empty_inter

    pred_mask = torch.isin(pred_pair_ids, common_ids)
    tgt_mask = torch.isin(tgt_pair_ids, common_ids)
    pred_ids = pred_pair_ids[pred_mask]
    pred_stems = pred_pair_stem[pred_mask]
    tgt_ids = tgt_pair_ids[tgt_mask]
    tgt_stems = tgt_pair_stem[tgt_mask]
    if pred_ids.numel() == 0 or tgt_ids.numel() == 0:
        empty_idx = pred_lengths.new_empty((0,), dtype=torch.long)
        empty_inter = pred_lengths.new_empty((0,), dtype=torch.float32)
        return empty_idx, empty_idx, empty_inter

    order_pred = torch.argsort(pred_ids)
    order_tgt = torch.argsort(tgt_ids)
    pred_ids = pred_ids[order_pred]
    pred_stems = pred_stems[order_pred]
    tgt_ids = tgt_ids[order_tgt]
    tgt_stems = tgt_stems[order_tgt]

    pred_count = int(pred_lengths.shape[0])
    tgt_count = int(tgt_lengths.shape[0])
    flat = tgt_stems * pred_count + pred_stems
    uniq_flat, inter_counts = torch.unique(flat, return_counts=True)
    gi = uniq_flat // pred_count
    pj = uniq_flat % pred_count
    inter = inter_counts.to(dtype=torch.float32)
    pred_len = pred_lengths[pj].to(dtype=torch.float32)
    tgt_len = tgt_lengths[gi].to(dtype=torch.float32)
    denom = pred_len + tgt_len - inter
    coef = torch.where(denom > 0, inter / denom, torch.zeros_like(inter))
    base_score = inter if weight_by_length else torch.ones_like(inter, dtype=torch.float32)
    score_mat = pred_lengths.new_full((pred_count, tgt_count), fill_value=-1.0, dtype=torch.float32)
    tgt_len_f = tgt_lengths[gi].to(dtype=torch.float32)
    score_mat[pj, gi] = base_score + (coef * MATCH_TIEBREAK_OVERLAP) + (tgt_len_f * MATCH_TIEBREAK_SIZE)
    inter_mat = pred_lengths.new_zeros((pred_count, tgt_count), dtype=torch.float32)
    inter_mat[pj, gi] = inter

    matched_pred, matched_tgt = bipartite_match(score_mat)
    if matched_pred.numel() == 0:
        empty_idx = pred_lengths.new_empty((0,), dtype=torch.long)
        empty_inter = pred_lengths.new_empty((0,), dtype=torch.float32)
        return empty_idx, empty_idx, empty_inter
    matched_inter = inter_mat[matched_pred, matched_tgt]
    return matched_pred, matched_tgt, matched_inter


def stem_confusion_from_arrays(
    pred_lengths: Tensor,
    pred_pair_ids: Tensor,
    pred_pair_stem: Tensor,
    tgt_lengths: Tensor,
    tgt_pair_ids: Tensor,
    tgt_pair_stem: Tensor,
) -> Tensor:
    device = pred_lengths.device

    if pred_lengths.numel() == 0 and tgt_lengths.numel() == 0:
        return torch.zeros((2, 2), dtype=torch.float32, device=device)

    pred_total = pred_lengths.sum().to(dtype=torch.float32)
    tgt_total = tgt_lengths.sum().to(dtype=torch.float32)
    if pred_lengths.numel() == 0:
        zero = tgt_total.new_zeros(())
        return make_confusion_matrix(zero, zero, tgt_total)
    if tgt_lengths.numel() == 0:
        zero = pred_total.new_zeros(())
        return make_confusion_matrix(zero, pred_total, zero)
    _, _, matched_inter = stem_match_indices(
        pred_lengths,
        pred_pair_ids,
        pred_pair_stem,
        tgt_lengths,
        tgt_pair_ids,
        tgt_pair_stem,
    )
    tp_len = matched_inter.sum().to(dtype=torch.float32)
    fn_len = tgt_total - tp_len
    fp_len = pred_total - tp_len
    return make_confusion_matrix(tp_len, fp_len, fn_len)


def stem_pairs_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return context.stem_confusion


def stem_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _stem_topology_confusion(context)


def stem_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = stem_confusion(context)
    return f1_from_confusion(cm, context.device)


def stem_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = stem_confusion(context)
    return precision_from_confusion(cm, context.device)


def stem_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = stem_confusion(context)
    return recall_from_confusion(cm, context.device)


def stem_pairs_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = stem_pairs_confusion(context)
    return f1_from_confusion(cm, context.device)


def stem_pairs_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = stem_pairs_confusion(context)
    return precision_from_confusion(cm, context.device)


def stem_pairs_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = stem_pairs_confusion(context)
    return recall_from_confusion(cm, context.device)


def stem_precision_recall_curve(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor, Tensor]:
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
    labels = stem_labels_from_gt(
        cand_lengths,
        gt_lengths,
        cand_pair_ids=cand_pair_ids,
        cand_pair_stem=cand_pair_stem,
        gt_pair_ids=context.target_stem_pair_ids,
        gt_pair_stem=context.target_stem_pair_stem,
    )

    scores_t = candidate_scores.to(dtype=torch.float32)
    labels_t = labels.to(dtype=torch.int64)
    return binary_precision_recall_curve(scores_t, labels_t)


def stem_candidate_segments(scores: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    min_len = _STEM_PR_MIN_LEN
    score_agg = _STEM_PR_SCORE_AGG
    pairs = contact_map_to_pairs(scores, unsafe=True, threshold=0.0)
    if pairs.numel() == 0:
        empty_scores = scores.new_empty((0,), dtype=torch.float32)
        empty_idx = scores.new_empty((0,), dtype=torch.long)
        return empty_scores, empty_idx, empty_idx, empty_idx, empty_idx, empty_idx
    start_i, start_j, lengths = pairs_to_stem_segment_arrays(pairs)
    if start_i.numel() == 0:
        empty_scores = scores.new_empty((0,), dtype=torch.float32)
        empty_idx = scores.new_empty((0,), dtype=torch.long)
        return empty_scores, empty_idx, empty_idx, empty_idx, empty_idx, empty_idx

    keep = lengths >= min_len
    keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)
    if keep_idx.numel() == 0:
        empty_scores = scores.new_empty((0,), dtype=torch.float32)
        empty_idx = scores.new_empty((0,), dtype=torch.long)
        return empty_scores, empty_idx, empty_idx, empty_idx, empty_idx, empty_idx
    start_i = start_i[keep_idx]
    start_j = start_j[keep_idx]
    lengths = lengths[keep_idx]

    num_stems = int(start_i.numel())
    total_pairs = lengths.sum()
    stem_ids = torch.repeat_interleave(torch.arange(num_stems, device=scores.device), lengths)
    prefix = lengths.cumsum(0)
    offsets = torch.arange(total_pairs, device=scores.device) - torch.repeat_interleave(prefix - lengths, lengths)
    pair_i = start_i[stem_ids] + offsets
    pair_j = start_j[stem_ids] - offsets
    pair_scores = scores[pair_i, pair_j].to(dtype=torch.float32)

    if score_agg == "mean":
        sums = torch.zeros(num_stems, device=scores.device, dtype=pair_scores.dtype)
        sums.scatter_add_(0, stem_ids, pair_scores)
        cand_scores = sums / lengths.to(dtype=pair_scores.dtype)
    else:
        fill_value = pair_scores.max()
        cand_scores = fill_value.expand(num_stems).clone()
        cand_scores.scatter_reduce_(0, stem_ids, pair_scores, reduce="amin", include_self=True)

    base = int(scores.shape[0]) + 1
    pair_ids = pair_i * base + pair_j
    return (
        cand_scores,
        start_i.to(dtype=torch.long),
        start_j.to(dtype=torch.long),
        lengths.to(dtype=torch.long),
        pair_ids.to(dtype=torch.long),
        stem_ids.to(dtype=torch.long),
    )


def stem_labels_from_gt(
    cand_lengths: Tensor,
    gt_lengths: Tensor,
    *,
    cand_pair_ids: Tensor,
    cand_pair_stem: Tensor,
    gt_pair_ids: Tensor,
    gt_pair_stem: Tensor,
    gt_mask: Optional[Tensor] = None,
) -> Tensor:
    if cand_lengths.numel() == 0:
        return cand_lengths.new_empty((0,), dtype=torch.int64)
    if gt_lengths.numel() == 0:
        return cand_lengths.new_zeros((cand_lengths.shape[0],), dtype=torch.int64)

    if gt_mask is not None:
        gt_lengths, gt_pair_ids, gt_pair_stem = subset_stem_data(gt_lengths, gt_pair_ids, gt_pair_stem, gt_mask)
        if gt_lengths.numel() == 0:
            return cand_lengths.new_zeros((cand_lengths.shape[0],), dtype=torch.int64)

    matched_pred, _, _ = stem_match_indices(
        cand_lengths,
        cand_pair_ids,
        cand_pair_stem,
        gt_lengths,
        gt_pair_ids,
        gt_pair_stem,
    )
    labels = cand_lengths.new_zeros((cand_lengths.shape[0],), dtype=torch.int64)
    if matched_pred.numel():
        labels[matched_pred] = 1
    return labels
