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

from typing import Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics.functional.classification import binary_precision_recall_curve

from ....utils.rna.secondary_structure import EndSide, HelixSegment, LoopSegmentType, LoopType, PseudoknotType

# Tiebreakers for bipartite matching: when Jaccard scores tie, prefer
# higher overlap count (MATCH_TIEBREAK_OVERLAP), then larger target
# size (MATCH_TIEBREAK_SIZE) to stabilize the Hungarian assignment.
MATCH_TIEBREAK_OVERLAP: float = 1e-6
MATCH_TIEBREAK_SIZE: float = 1e-9


def loop_taxonomy_compatible(pred_loop, tgt_loop) -> bool:
    pred_pseudoknot_type = pred_loop.pseudoknot_type
    tgt_pseudoknot_type = tgt_loop.pseudoknot_type
    if pred_pseudoknot_type != tgt_pseudoknot_type and PseudoknotType.UNKNOWN not in (
        pred_pseudoknot_type,
        tgt_pseudoknot_type,
    ):
        return False
    if pred_loop.role is not None and tgt_loop.role is not None and pred_loop.role != tgt_loop.role:
        return False
    return True


def _segment_side_value(segment) -> int:
    if segment.side is None:
        return -1
    return 0 if segment.side is EndSide.FIVE_PRIME else 1


def segment_overlap_components(
    pred_segments: Sequence, target_segments: Sequence, *, device: torch.device, enforce_end_side: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    pred_total = len(pred_segments)
    target_total = len(target_segments)
    if pred_total == 0 or target_total == 0:
        pred_sizes = torch.empty((pred_total,), device=device, dtype=torch.float32)
        target_sizes = torch.empty((target_total,), device=device, dtype=torch.float32)
        overlap = torch.empty((pred_total, target_total), device=device, dtype=torch.float32)
        valid = torch.zeros((pred_total, target_total), device=device, dtype=torch.bool)
        return pred_sizes, target_sizes, overlap, valid

    pred_start = torch.tensor([int(seg.start) for seg in pred_segments], device=device, dtype=torch.long)
    pred_stop = torch.tensor([int(seg.stop) for seg in pred_segments], device=device, dtype=torch.long)
    tgt_start = torch.tensor([int(seg.start) for seg in target_segments], device=device, dtype=torch.long)
    tgt_stop = torch.tensor([int(seg.stop) for seg in target_segments], device=device, dtype=torch.long)
    pred_sizes = (pred_stop - pred_start + 1).to(dtype=torch.float32)
    target_sizes = (tgt_stop - tgt_start + 1).to(dtype=torch.float32)

    start = torch.maximum(pred_start[:, None], tgt_start[None, :])
    stop = torch.minimum(pred_stop[:, None], tgt_stop[None, :])
    overlap = (stop - start + 1).clamp(min=0).to(dtype=torch.float32)
    valid = overlap > 0
    if enforce_end_side:
        pred_side = torch.tensor([_segment_side_value(seg) for seg in pred_segments], device=device, dtype=torch.long)[
            :, None
        ]
        tgt_side = torch.tensor([_segment_side_value(seg) for seg in target_segments], device=device, dtype=torch.long)[
            None, :
        ]
        valid = valid & ((pred_side < 0) | (tgt_side < 0) | (pred_side == tgt_side))
    return pred_sizes, target_sizes, overlap, valid


def _interval_masks(
    owner_ids: Tensor, starts: Tensor, stops: Tensor, *, count: int, length: int, device: torch.device
) -> Tensor:
    if count <= 0:
        return torch.zeros((0, max(length, 0)), dtype=torch.bool, device=device)
    if length <= 0:
        return torch.zeros((count, 0), dtype=torch.bool, device=device)
    if owner_ids.numel() == 0:
        return torch.zeros((count, length), dtype=torch.bool, device=device)
    diff = torch.zeros((count, length + 1), dtype=torch.int32, device=device)
    stride = length + 1
    flat = diff.view(-1)
    starts = starts.clamp(min=0, max=length)
    stops = stops.clamp(min=-1, max=length - 1)
    valid = starts <= stops
    if not bool(valid.any().item()):
        return torch.zeros((count, length), dtype=torch.bool, device=device)
    owner_ids = owner_ids[valid]
    starts = starts[valid]
    stops = stops[valid]
    ones = torch.ones_like(starts, dtype=flat.dtype)
    flat.scatter_add_(0, owner_ids * stride + starts, ones)
    flat.scatter_add_(0, owner_ids * stride + (stops + 1), -ones)
    return torch.cumsum(diff[:, :-1], dim=1) > 0


def _loop_nucleotide_masks(loops: Sequence, *, length: int, device: torch.device) -> Tensor:
    loop_count = len(loops)
    if loop_count == 0:
        return torch.zeros((0, max(length, 0)), dtype=torch.bool, device=device)
    owner: list[int] = []
    starts: list[int] = []
    stops: list[int] = []
    for idx, loop in enumerate(loops):
        for span in loop.spans:
            owner.append(idx)
            starts.append(int(span.start))
            stops.append(int(span.stop))
    if not owner:
        return torch.zeros((loop_count, max(length, 0)), dtype=torch.bool, device=device)
    owner_t = torch.tensor(owner, dtype=torch.long, device=device)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device)
    stops_t = torch.tensor(stops, dtype=torch.long, device=device)
    return _interval_masks(owner_t, starts_t, stops_t, count=loop_count, length=length, device=device)


def _loop_taxonomy_compatibility_matrix(
    pred_loops: Sequence, target_loops: Sequence, *, device: torch.device
) -> Tensor:
    pred_count = len(pred_loops)
    tgt_count = len(target_loops)
    if pred_count == 0 or tgt_count == 0:
        return torch.zeros((pred_count, tgt_count), dtype=torch.bool, device=device)

    pred_pk = [loop.pseudoknot_type for loop in pred_loops]
    tgt_pk = [loop.pseudoknot_type for loop in target_loops]
    pk_mapping: dict[object, int] = {}
    for value in (*pred_pk, *tgt_pk):
        if value not in pk_mapping:
            pk_mapping[value] = len(pk_mapping)
    pred_pk_ids = torch.tensor([pk_mapping[value] for value in pred_pk], dtype=torch.long, device=device)
    tgt_pk_ids = torch.tensor([pk_mapping[value] for value in tgt_pk], dtype=torch.long, device=device)
    pk_compat = pred_pk_ids[:, None] == tgt_pk_ids[None, :]
    unknown_id = pk_mapping.get(PseudoknotType.UNKNOWN, -1)
    if unknown_id >= 0:
        pk_compat = pk_compat | (pred_pk_ids[:, None] == unknown_id) | (tgt_pk_ids[None, :] == unknown_id)

    pred_roles = [loop.role for loop in pred_loops]
    tgt_roles = [loop.role for loop in target_loops]
    role_mapping: dict[object, int] = {}
    for value in (*pred_roles, *tgt_roles):
        if value not in role_mapping:
            role_mapping[value] = len(role_mapping)
    pred_role_ids = torch.tensor([role_mapping[value] for value in pred_roles], dtype=torch.long, device=device)
    tgt_role_ids = torch.tensor([role_mapping[value] for value in tgt_roles], dtype=torch.long, device=device)
    role_compat = pred_role_ids[:, None] == tgt_role_ids[None, :]
    none_id = role_mapping.get(None, -1)
    if none_id >= 0:
        role_compat = role_compat | (pred_role_ids[:, None] == none_id) | (tgt_role_ids[None, :] == none_id)

    return pk_compat & role_compat


def loop_overlap_components(
    pred_loops: Sequence, target_loops: Sequence, *, length: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    pred_count = len(pred_loops)
    tgt_count = len(target_loops)
    pred_sizes = torch.tensor([loop.size for loop in pred_loops], device=device, dtype=torch.float32)
    target_sizes = torch.tensor([loop.size for loop in target_loops], device=device, dtype=torch.float32)
    if pred_count == 0 or tgt_count == 0:
        overlap = torch.empty((pred_count, tgt_count), device=device, dtype=torch.float32)
        valid = torch.zeros((pred_count, tgt_count), device=device, dtype=torch.bool)
        return pred_sizes, target_sizes, overlap, valid

    pred_mask = _loop_nucleotide_masks(pred_loops, length=length, device=device).to(dtype=torch.float32)
    tgt_mask = _loop_nucleotide_masks(target_loops, length=length, device=device).to(dtype=torch.float32)
    overlap = pred_mask @ tgt_mask.T
    taxonomy = _loop_taxonomy_compatibility_matrix(pred_loops, target_loops, device=device)
    valid = (overlap > 0) & taxonomy
    return pred_sizes, target_sizes, overlap, valid


def anchor_pairs_score(pred_anchors: Sequence[tuple[int, int]], target_anchors: Sequence[tuple[int, int]]) -> float:
    if not pred_anchors and not target_anchors:
        return 1.0
    if not pred_anchors or not target_anchors:
        return 0.0

    if len(pred_anchors) != len(target_anchors):
        return 0.0

    pred_norm = [(min(i, j), max(i, j)) for i, j in pred_anchors]
    target_norm = [(min(i, j), max(i, j)) for i, j in target_anchors]
    return 1.0 if pred_norm == target_norm else 0.0


def pair_overlap_score_matrix(
    pred_lengths: Tensor,
    pred_pair_ids: Tensor,
    pred_pair_owner: Tensor,
    tgt_lengths: Tensor,
    tgt_pair_ids: Tensor,
    tgt_pair_owner: Tensor,
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    pred_count = int(pred_lengths.shape[0])
    tgt_count = int(tgt_lengths.shape[0])
    score_mat = torch.full((pred_count, tgt_count), -1.0, device=device, dtype=torch.float32)
    inter_mat = torch.zeros((pred_count, tgt_count), device=device, dtype=torch.float32)
    if pred_count == 0 or tgt_count == 0:
        return score_mat, inter_mat
    if pred_pair_ids.numel() == 0 or tgt_pair_ids.numel() == 0:
        return score_mat, inter_mat

    def _unique_ids_and_owner(pair_ids: Tensor, pair_owner: Tensor) -> tuple[Tensor, Tensor]:
        order = torch.argsort(pair_ids)
        ids_sorted = pair_ids[order]
        owner_sorted = pair_owner[order]
        uniq_ids, counts = torch.unique_consecutive(ids_sorted, return_counts=True)
        if uniq_ids.numel() == 0:
            return uniq_ids, owner_sorted
        starts = counts.cumsum(0) - counts
        return uniq_ids, owner_sorted[starts]

    pred_ids, pred_owner = _unique_ids_and_owner(pred_pair_ids, pred_pair_owner)
    tgt_ids, tgt_owner = _unique_ids_and_owner(tgt_pair_ids, tgt_pair_owner)
    if pred_ids.numel() == 0 or tgt_ids.numel() == 0:
        return score_mat, inter_mat

    pred_mask = torch.isin(pred_ids, tgt_ids)
    if not bool(pred_mask.any().item()):
        return score_mat, inter_mat
    pred_ids = pred_ids[pred_mask]
    pred_owner = pred_owner[pred_mask]
    tgt_pos = torch.searchsorted(tgt_ids, pred_ids)
    valid = (tgt_pos < tgt_ids.numel()) & (tgt_ids[tgt_pos] == pred_ids)
    if not bool(valid.any().item()):
        return score_mat, inter_mat
    pred_owner = pred_owner[valid]
    tgt_owner = tgt_owner[tgt_pos[valid]]

    flat = tgt_owner * pred_count + pred_owner
    uniq_flat, inter_counts = torch.unique(flat, return_counts=True)
    gi = uniq_flat // pred_count
    pj = uniq_flat % pred_count
    inter = inter_counts.to(dtype=torch.float32)
    pred_len = pred_lengths[pj].to(dtype=torch.float32)
    tgt_len = tgt_lengths[gi].to(dtype=torch.float32)
    denom = pred_len + tgt_len - inter
    coef = torch.where(denom > 0, inter / denom, torch.zeros_like(inter))
    score_mat[pj, gi] = coef
    inter_mat[pj, gi] = inter
    return score_mat, inter_mat


def helix_score_matrix(
    pred_helices: Sequence[HelixSegment], tgt_helices: Sequence[HelixSegment], device: torch.device
) -> Tensor:
    pred_count = len(pred_helices)
    tgt_count = len(tgt_helices)
    if pred_count == 0 or tgt_count == 0:
        return torch.empty((pred_count, tgt_count), device=device, dtype=torch.float32)

    max_idx = -1
    for helix in (*pred_helices, *tgt_helices):
        for i, j in helix.pairs:
            i = int(i)
            j = int(j)
            if i > j:
                i, j = j, i
            if i > max_idx:
                max_idx = i
            if j > max_idx:
                max_idx = j
    base = max_idx + 1 if max_idx >= 0 else 1

    pred_lengths = torch.tensor([len(helix) for helix in pred_helices], device=device, dtype=torch.long)
    tgt_lengths = torch.tensor([len(helix) for helix in tgt_helices], device=device, dtype=torch.long)

    def _pair_ids_and_owner(helices: Sequence[HelixSegment]) -> tuple[Tensor, Tensor]:
        pair_list: list[tuple[int, int]] = []
        owner_list: list[int] = []
        for idx, helix in enumerate(helices):
            for i, j in helix.pairs:
                i = int(i)
                j = int(j)
                if i > j:
                    i, j = j, i
                pair_list.append((i, j))
                owner_list.append(idx)
        if not pair_list:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty
        pair_tensor = torch.tensor(pair_list, device=device, dtype=torch.long)
        pair_ids = pair_tensor[:, 0] * base + pair_tensor[:, 1]
        owner_tensor = torch.tensor(owner_list, device=device, dtype=torch.long)
        return pair_ids, owner_tensor

    pred_pair_ids, pred_owner = _pair_ids_and_owner(pred_helices)
    tgt_pair_ids, tgt_owner = _pair_ids_and_owner(tgt_helices)

    score_mat, inter_mat = pair_overlap_score_matrix(
        pred_lengths,
        pred_pair_ids,
        pred_owner,
        tgt_lengths,
        tgt_pair_ids,
        tgt_owner,
        device=device,
    )
    if score_mat.numel() == 0:
        return score_mat
    valid = score_mat >= 0
    if valid.any():
        score_mat = score_mat.clone()
        score_mat[valid] += inter_mat[valid] * MATCH_TIEBREAK_OVERLAP
        tgt_size_mat = tgt_lengths.unsqueeze(0).to(dtype=torch.float32).expand_as(score_mat)
        score_mat[valid] += tgt_size_mat[valid] * MATCH_TIEBREAK_SIZE
    return score_mat


def loop_overlap_len(pred_spans: Sequence, target_spans: Sequence) -> int:
    i = 0
    j = 0
    overlap = 0
    while i < len(pred_spans) and j < len(target_spans):
        a = pred_spans[i]
        b = target_spans[j]
        start = max(int(a.start), int(b.start))
        stop = min(int(a.stop), int(b.stop))
        if start <= stop:
            overlap += stop - start + 1
        if a.stop < b.stop:
            i += 1
        else:
            j += 1
    return overlap


def segment_type_from_loop(loop_type: LoopType) -> LoopSegmentType:
    mapping = {
        LoopType.HAIRPIN: LoopSegmentType.HAIRPIN,
        LoopType.BULGE: LoopSegmentType.BULGE,
        LoopType.INTERNAL: LoopSegmentType.INTERNAL,
        LoopType.MULTILOOP: LoopSegmentType.BRANCH,
        LoopType.EXTERNAL: LoopSegmentType.EXTERNAL,
    }
    segment_type = mapping.get(loop_type)
    if segment_type is None:
        raise ValueError(f"Unsupported loop type: {loop_type}")
    return segment_type


def segments_from_topology(topology) -> list:
    return topology.loop_segments()


def make_confusion_matrix(tp: Tensor, fp: Tensor, fn: Tensor, tn: Tensor | None = None) -> Tensor:
    """Build a 2x2 binary confusion matrix in standard layout: ``[[TN, FP], [FN, TP]]``."""
    if tn is None:
        tn = tp.new_zeros(())
    return torch.stack([torch.stack([tn, fp]), torch.stack([fn, tp])])


def confusion_from_items(pred_items: Tensor, target_items: Tensor, device: torch.device) -> Tensor:
    if pred_items.ndim == 1:
        all_items = torch.cat([pred_items, target_items])
        _, counts = torch.unique(all_items, return_counts=True)
    else:
        all_items = torch.cat([pred_items, target_items], dim=0)
        _, counts = torch.unique(all_items, return_counts=True, dim=0)
    tp = (counts == 2).sum().to(dtype=torch.float32)
    pred_total = torch.tensor(pred_items.shape[0], device=device, dtype=torch.float32)
    target_total = torch.tensor(target_items.shape[0], device=device, dtype=torch.float32)
    fp = pred_total - tp
    fn = target_total - tp
    return make_confusion_matrix(tp, fp, fn)


def pair_exact_match(pred_pairs: Tensor, target_pairs: Tensor, device: torch.device | None = None) -> Tensor:
    if device is None:
        device = pred_pairs.device
    target_total = int(target_pairs.shape[0])
    if target_total == 0:
        return torch.tensor(float("nan"), device=device, dtype=torch.float32)
    cm = confusion_from_items(pred_pairs, target_pairs, device)
    tp_val = int(cm[1, 1].item())
    pred_total = int(pred_pairs.shape[0])
    return cm.new_full((), 1.0 if tp_val == pred_total == target_total else 0.0)


def pair_error_rate(pred_pairs: Tensor, target_pairs: Tensor, device: torch.device | None = None) -> Tensor:
    if device is None:
        device = pred_pairs.device
    target_total = int(target_pairs.shape[0])
    if target_total == 0:
        return torch.tensor(float("nan"), device=device, dtype=torch.float32)
    cm = confusion_from_items(pred_pairs, target_pairs, device)
    fp = cm[0, 1]
    fn = cm[1, 0]
    denom = cm[1, 1] + fp + fn
    return torch.where(denom > 0, (fp + fn) / denom, cm.new_zeros(()))


def bipartite_match(scores: Tensor) -> Tuple[Tensor, Tensor]:
    if scores.numel() == 0:
        empty = scores.new_empty((0,), dtype=torch.long)
        return empty, empty
    valid_mask = scores >= 0
    row_idx = torch.nonzero(valid_mask.any(dim=1), as_tuple=False).view(-1)
    col_idx = torch.nonzero(valid_mask.any(dim=0), as_tuple=False).view(-1)
    if row_idx.numel() == 0 or col_idx.numel() == 0:
        empty = scores.new_empty((0,), dtype=torch.long)
        return empty, empty
    scores = scores[row_idx][:, col_idx]
    num_rows, num_cols = scores.shape
    if num_rows == 0 or num_cols == 0:
        empty = scores.new_empty((0,), dtype=torch.long)
        return empty, empty
    device = scores.device
    scores_cpu = scores.detach()
    if scores_cpu.device.type != "cpu":
        scores_cpu = scores_cpu.to(device="cpu")
    if scores_cpu.dtype != torch.float32:
        scores_cpu = scores_cpu.to(dtype=torch.float32)
    scores_cpu = scores_cpu.numpy()
    total_cols = num_cols + num_rows
    extended = np.zeros((num_rows, total_cols), dtype=np.float32)
    extended[:, :num_cols] = scores_cpu
    max_score = float(np.nanmax(extended)) if extended.size else 0.0
    if not np.isfinite(max_score) or max_score < 0.0:
        max_score = 0.0
    cost = max_score - extended
    row_ind, col_ind = linear_sum_assignment(cost)
    valid = col_ind < num_cols
    if not np.any(valid):
        empty = scores.new_empty((0,), dtype=torch.long)
        return empty, empty
    matched_rows = row_idx[torch.as_tensor(row_ind[valid], device=device, dtype=torch.long)]
    matched_cols = col_idx[torch.as_tensor(col_ind[valid], device=device, dtype=torch.long)]
    return matched_rows, matched_cols


def pack_neighbors(neighbors: Sequence[Sequence[int]], device: torch.device) -> Tuple[Tensor, Tensor]:
    """Pack variable-length neighbor lists into (lengths, padded) tensors."""
    count = len(neighbors)
    lengths = torch.tensor([len(item) for item in neighbors], dtype=torch.long, device=device)
    max_len = int(lengths.max().item()) if count and lengths.numel() else 0
    if max_len == 0:
        return lengths, torch.empty((count, 0), dtype=torch.long, device=device)
    packed = torch.full((count, max_len), -1, dtype=torch.long, device=device)
    for idx, item in enumerate(neighbors):
        if not item:
            continue
        vals = torch.as_tensor(item, dtype=torch.long, device=device)
        packed[idx, : vals.numel()] = vals
    return lengths, packed


def pairs_precision_recall_curve(scores: Tensor, target_pairs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    device = scores.device
    n = int(scores.shape[0])
    tri = torch.triu_indices(n, n, offset=1, device=device)
    scores_t = scores[tri[0], tri[1]].to(dtype=torch.float32)
    labels = torch.zeros(scores_t.shape[0], device=device, dtype=torch.int64)
    if target_pairs.numel():
        low = torch.minimum(target_pairs[:, 0], target_pairs[:, 1]).to(dtype=torch.long)
        high = torch.maximum(target_pairs[:, 0], target_pairs[:, 1]).to(dtype=torch.long)
        row_offset = low * (2 * n - low - 1) // 2
        idx = row_offset + (high - low - 1)
        labels[idx] = 1
    return binary_precision_recall_curve(scores_t, labels)


def _normalized_diff(a: float | int, b: float | int) -> float:
    denom = float(max(a, b))
    if denom == 0.0:
        return 0.0
    return abs(float(a) - float(b)) / denom


def loop_substitution_cost(pred_loop, tgt_loop) -> float:
    if pred_loop.kind != tgt_loop.kind:
        return 1.0
    components: list[float] = []
    components.append(_normalized_diff(pred_loop.size, tgt_loop.size))
    if pred_loop.branch_count or tgt_loop.branch_count:
        components.append(_normalized_diff(pred_loop.branch_count, tgt_loop.branch_count))
    if pred_loop.asymmetry or tgt_loop.asymmetry:
        components.append(_normalized_diff(pred_loop.asymmetry, tgt_loop.asymmetry))
    if (
        pred_loop.pseudoknot_type is not None
        and tgt_loop.pseudoknot_type is not None  # noqa: W503
        and pred_loop.pseudoknot_type is not PseudoknotType.UNKNOWN  # noqa: W503
        and tgt_loop.pseudoknot_type is not PseudoknotType.UNKNOWN  # noqa: W503
        and pred_loop.pseudoknot_type != tgt_loop.pseudoknot_type  # noqa: W503
    ):
        components.append(1.0)
    if pred_loop.role is not None and tgt_loop.role is not None:
        components.append(0.0 if pred_loop.role == tgt_loop.role else 1.0)
    if not components:
        return 0.0
    return float(sum(components) / len(components))


def segment_length_cost(a, b) -> float:
    return _normalized_diff(len(a), len(b))


def _edge_label_cost(a, b) -> float:
    return 0.0 if a == b else 1.0


def approximate_graph_edit_distance(
    pred_nodes: Sequence,
    tgt_nodes: Sequence,
    pred_edges: Sequence[tuple[int, int, object]],
    tgt_edges: Sequence[tuple[int, int, object]],
    *,
    node_cost_fn,
    edge_cost_fn=_edge_label_cost,
    node_ins_cost: float = 1.0,
    node_del_cost: float = 1.0,
    edge_ins_cost: float = 1.0,
    edge_del_cost: float = 1.0,
    device: torch.device | None = None,
) -> Tensor:
    pred_count = len(pred_nodes)
    tgt_count = len(tgt_nodes)
    size = pred_count + tgt_count
    if size == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    cost: np.ndarray = np.zeros((size, size), dtype=np.float32)
    if pred_count and tgt_count:
        for i, pred_node in enumerate(pred_nodes):
            for j, tgt_node in enumerate(tgt_nodes):
                cost[i, j] = float(node_cost_fn(pred_node, tgt_node))
    if pred_count:
        cost[:pred_count, tgt_count:] = float(node_del_cost)
    if tgt_count:
        cost[pred_count:, :tgt_count] = float(node_ins_cost)

    row_ind, col_ind = linear_sum_assignment(cost)
    node_cost = float(cost[row_ind, col_ind].sum())

    mapping: dict[int, int] = {}
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if row < pred_count and col < tgt_count:
            mapping[row] = col

    target_edges: dict[tuple[int, int], list[object]] = {}
    for u, v, label in tgt_edges:
        target_edges.setdefault((u, v), []).append(label)

    edge_cost = 0.0
    for u, v, label in pred_edges:
        mapped_u = mapping.get(u)
        mapped_v = mapping.get(v)
        if mapped_u is None or mapped_v is None:
            edge_cost += edge_del_cost
            continue
        key = (mapped_u, mapped_v)
        labels = target_edges.get(key)
        if not labels:
            edge_cost += edge_del_cost
            continue
        if label in labels:
            labels.remove(label)
            if not labels:
                target_edges.pop(key, None)
            continue
        other = labels.pop()
        edge_cost += float(edge_cost_fn(label, other))
        if not labels:
            target_edges.pop(key, None)

    if target_edges:
        remaining = sum(len(labels) for labels in target_edges.values())
        edge_cost += edge_ins_cost * float(remaining)

    total = node_cost + edge_cost
    denom = float(pred_count + tgt_count + len(pred_edges) + len(tgt_edges))
    if denom <= 0.0:
        denom = 1.0
    return torch.tensor(total / denom, device=device, dtype=torch.float32)


def f1_from_confusion(cm: Tensor, device: torch.device) -> Tensor:
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    target_pos = tp + fn
    if target_pos == 0:
        return torch.tensor(float("nan"), device=device, dtype=cm.dtype)
    denom = (2 * tp) + fp + fn
    zero = torch.zeros((), device=device, dtype=cm.dtype)
    return torch.where(denom > 0, (2 * tp) / denom, zero)


def precision_from_confusion(cm: Tensor, device: torch.device) -> Tensor:
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    target_pos = tp + fn
    if target_pos == 0:
        return torch.tensor(float("nan"), device=device, dtype=cm.dtype)
    denom = tp + fp
    zero = torch.zeros((), device=device, dtype=cm.dtype)
    return torch.where(denom > 0, tp / denom, zero)


def recall_from_confusion(cm: Tensor, device: torch.device) -> Tensor:
    tp = cm[1, 1]
    fn = cm[1, 0]
    target_pos = tp + fn
    if target_pos == 0:
        return torch.tensor(float("nan"), device=device, dtype=cm.dtype)
    denom = tp + fn
    zero = torch.zeros((), device=device, dtype=cm.dtype)
    return torch.where(denom > 0, tp / denom, zero)
