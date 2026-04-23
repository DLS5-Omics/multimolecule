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

from typing import TYPE_CHECKING, Sequence, Tuple

import torch
from torch import Tensor

from .common import (
    MATCH_TIEBREAK_OVERLAP,
    MATCH_TIEBREAK_SIZE,
    bipartite_match,
    f1_from_confusion,
    loop_overlap_len,
    pack_neighbors,
    precision_from_confusion,
    recall_from_confusion,
)
from .loops import (
    loop_anchor_adjacency_score,
    loop_anchor_coaxial_score,
    loop_anchor_order_score_any,
    loop_anchor_pairs_score_any,
    loop_anchor_wiring_score,
)

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def loop_helix_edges_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.loop_helix_edges_confusion
    return f1_from_confusion(cm, context.device)


def loop_helix_edges_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.loop_helix_edges_confusion
    return precision_from_confusion(cm, context.device)


def loop_helix_edges_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.loop_helix_edges_confusion
    return recall_from_confusion(cm, context.device)


def _adjacency_score_matrix(
    pred_adj: Sequence[Sequence[int]], tgt_adj: Sequence[Sequence[int]], opposite_score: Tensor, device: torch.device
) -> Tensor:
    pred_count = len(pred_adj)
    tgt_count = len(tgt_adj)
    if pred_count == 0 or tgt_count == 0:
        return torch.empty((pred_count, tgt_count), device=device, dtype=torch.float32)
    score_mat = torch.zeros((pred_count, tgt_count), dtype=torch.float32, device=device)
    scores = opposite_score
    if scores.device != device:
        scores = scores.to(device=device)
    if scores.dtype != torch.float32:
        scores = scores.to(dtype=torch.float32)

    pred_lengths, pred_padded = pack_neighbors(pred_adj, device)
    tgt_lengths, tgt_padded = pack_neighbors(tgt_adj, device)
    pred_len = pred_lengths[:, None]
    tgt_len = tgt_lengths[None, :]
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    one = torch.tensor(1.0, device=device, dtype=torch.float32)

    both_empty = (pred_len == 0) & (tgt_len == 0)
    score_mat[both_empty] = 1.0

    mask_11 = (pred_len == 1) & (tgt_len == 1)
    if mask_11.any():
        pair_idx = torch.nonzero(mask_11, as_tuple=False)
        pi = pair_idx[:, 0]
        tj = pair_idx[:, 1]
        vals = scores[pred_padded[pi, 0], tgt_padded[tj, 0]]
        score_mat[pi, tj] = torch.where(vals >= 0, torch.minimum(vals, one), zero)

    mask_1n = (pred_len == 1) & (tgt_len > 1)
    if mask_1n.any():
        pair_idx = torch.nonzero(mask_1n, as_tuple=False)
        pi = pair_idx[:, 0]
        tj = pair_idx[:, 1]
        pred_ids = pred_padded[pi, 0]
        tgt_ids = tgt_padded[tj]
        valid = tgt_ids >= 0
        gather_tgt = torch.where(valid, tgt_ids, torch.zeros_like(tgt_ids))
        vals = scores[pred_ids[:, None], gather_tgt]
        vals = vals.masked_fill(~valid, neg_inf)
        max_val = vals.max(dim=1).values
        score_mat[pi, tj] = torch.where(max_val >= 0, torch.minimum(max_val, one), zero)

    mask_n1 = (pred_len > 1) & (tgt_len == 1)
    if mask_n1.any():
        pair_idx = torch.nonzero(mask_n1, as_tuple=False)
        pi = pair_idx[:, 0]
        tj = pair_idx[:, 1]
        pred_ids = pred_padded[pi]
        valid = pred_ids >= 0
        gather_pred = torch.where(valid, pred_ids, torch.zeros_like(pred_ids))
        vals = scores[gather_pred, tgt_padded[tj, 0][:, None]]
        vals = vals.masked_fill(~valid, neg_inf)
        max_val = vals.max(dim=1).values
        score_mat[pi, tj] = torch.where(max_val >= 0, torch.minimum(max_val, one), zero)

    mask_22 = (pred_len == 2) & (tgt_len == 2)
    if mask_22.any():
        pair_idx = torch.nonzero(mask_22, as_tuple=False)
        pi = pair_idx[:, 0]
        tj = pair_idx[:, 1]
        pred_ids = pred_padded[pi, :2]
        tgt_ids = tgt_padded[tj, :2]
        v00 = scores[pred_ids[:, 0], tgt_ids[:, 0]]
        v01 = scores[pred_ids[:, 0], tgt_ids[:, 1]]
        v10 = scores[pred_ids[:, 1], tgt_ids[:, 0]]
        v11 = scores[pred_ids[:, 1], tgt_ids[:, 1]]
        all_vals = torch.stack([v00, v01, v10, v11], dim=1)
        all_vals = all_vals.masked_fill(all_vals < 0, neg_inf)
        best_single = all_vals.max(dim=1).values
        diag = torch.where((v00 >= 0) & (v11 >= 0), v00 + v11, neg_inf)
        off_diag = torch.where((v01 >= 0) & (v10 >= 0), v01 + v10, neg_inf)
        best = torch.maximum(best_single, torch.maximum(diag, off_diag))
        best = torch.where(torch.isfinite(best), best / 2.0, zero)
        score_mat[pi, tj] = torch.minimum(best, one)

    mask_general = (pred_len > 1) & (tgt_len > 1) & (~mask_22)
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
                score_mat[i, j] = torch.minimum(sim_sum / denom, one)
    return score_mat


def _joint_node_score(
    base_score: Tensor,
    pred_adj: Sequence[Sequence[int]],
    tgt_adj: Sequence[Sequence[int]],
    opposite_score: Tensor,
    device: torch.device,
) -> Tensor:
    if base_score.numel() == 0:
        return base_score
    adj_score = _adjacency_score_matrix(pred_adj, tgt_adj, opposite_score, device)
    joint = torch.full_like(base_score, -1.0, dtype=torch.float32)
    valid = base_score >= 0
    if valid.any():
        joint_vals = (base_score[valid] + adj_score[valid]) / 2.0
        joint[valid] = torch.clamp(joint_vals, max=1.0)
    return joint


def _global_node_assignment(
    loop_score: Tensor, helix_score: Tensor, device: torch.device
) -> Tuple[dict[int, int], dict[int, int]]:
    pred_loops, tgt_loops = loop_score.shape
    pred_helices, tgt_helices = helix_score.shape
    total_pred = pred_loops + pred_helices
    total_tgt = tgt_loops + tgt_helices
    if total_pred == 0 or total_tgt == 0:
        return {}, {}
    score_mat = torch.full((total_pred, total_tgt), -1.0, device=device, dtype=torch.float32)
    if pred_loops and tgt_loops:
        score_mat[:pred_loops, :tgt_loops] = loop_score
    if pred_helices and tgt_helices:
        score_mat[pred_loops:, tgt_loops:] = helix_score
    matched_pred, matched_tgt = bipartite_match(score_mat)
    loop_map: dict[int, int] = {}
    helix_map: dict[int, int] = {}
    for pred_idx, tgt_idx in zip(matched_pred.tolist(), matched_tgt.tolist()):
        if pred_idx < pred_loops and tgt_idx < tgt_loops:
            loop_map[int(pred_idx)] = int(tgt_idx)
        elif pred_idx >= pred_loops and tgt_idx >= tgt_loops:
            helix_map[int(pred_idx - pred_loops)] = int(tgt_idx - tgt_loops)
    return loop_map, helix_map


def _count_consistent_edge_matches(
    pred_edges: Sequence[Tuple[int, int]],
    tgt_edges: Sequence[Tuple[int, int]],
    loop_map: dict[int, int],
    helix_map: dict[int, int],
) -> int:
    tgt_edges_set = set(tgt_edges)
    matches = 0
    for pred_loop, pred_helix in pred_edges:
        tgt_loop = loop_map.get(pred_loop)
        tgt_helix = helix_map.get(pred_helix)
        if tgt_loop is None or tgt_helix is None:
            continue
        if (tgt_loop, tgt_helix) in tgt_edges_set:
            matches += 1
    return matches


def _loop_score_matrix(
    pred_loops,
    tgt_loops,
    pred_helix_key_map: dict[tuple[int, int, int, int], int],
    tgt_helix_key_map: dict[tuple[int, int, int, int], int],
    pred_to_tgt: dict[int, int],
    pred_helix_edges: list[tuple[int, int, int]],
    tgt_helix_edges: list[tuple[int, int, int]],
    device: torch.device,
) -> Tensor:
    pred_count = len(pred_loops)
    tgt_count = len(tgt_loops)
    if pred_count == 0 or tgt_count == 0:
        return torch.empty((pred_count, tgt_count), device=device, dtype=torch.float32)
    score_mat = torch.full((pred_count, tgt_count), -1.0, device=device, dtype=torch.float32)
    for i, pred_loop in enumerate(pred_loops):
        for j, tgt_loop in enumerate(tgt_loops):
            anchor_score = loop_anchor_pairs_score_any(pred_loop, tgt_loop)
            if anchor_score <= 0.0:
                continue
            wiring_score = loop_anchor_wiring_score(
                pred_loop.kind,
                pred_loop,
                tgt_loop,
                pred_helix_key_map,
                tgt_helix_key_map,
                pred_to_tgt,
            )
            if wiring_score <= 0.0:
                continue
            order_score = loop_anchor_order_score_any(
                pred_loop,
                tgt_loop,
                pred_helix_key_map,
                tgt_helix_key_map,
                pred_to_tgt,
            )
            if order_score <= 0.0:
                continue
            adjacency_score = loop_anchor_adjacency_score(
                pred_loop,
                tgt_loop,
                pred_helix_key_map,
                tgt_helix_key_map,
                pred_to_tgt,
                pred_helix_edges,
                tgt_helix_edges,
            )
            if adjacency_score <= 0.0:
                continue
            coaxial_score = loop_anchor_coaxial_score(
                pred_loop,
                tgt_loop,
                pred_helix_key_map,
                tgt_helix_key_map,
                pred_to_tgt,
                pred_helix_edges,
                tgt_helix_edges,
            )
            if coaxial_score <= 0.0:
                continue
            overlap_len = loop_overlap_len(pred_loop.spans, tgt_loop.spans)
            if overlap_len <= 0:
                continue
            union_len = pred_loop.size + tgt_loop.size - overlap_len
            if union_len <= 0:
                continue
            jaccard = overlap_len / union_len
            if jaccard > 0.0:
                score_mat[i, j] = float(
                    jaccard + overlap_len * MATCH_TIEBREAK_OVERLAP + tgt_loop.size * MATCH_TIEBREAK_SIZE
                )
    return score_mat
