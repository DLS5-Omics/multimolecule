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

from collections import Counter
from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_confusion_matrix

from ....utils.rna.secondary_structure import LoopSegmentType, LoopType
from .common import (
    MATCH_TIEBREAK_OVERLAP,
    MATCH_TIEBREAK_SIZE,
    anchor_pairs_score,
    bipartite_match,
    loop_overlap_components,
    make_confusion_matrix,
    segment_overlap_components,
)
from .functional import binary_f1_score, binary_mcc, binary_precision, binary_recall, f1_from_pr

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def _loop_anchor_pairs_score(loop_type: LoopType, pred_loop, tgt_loop) -> float:
    pred_count = len(pred_loop.anchor_pairs)
    tgt_count = len(tgt_loop.anchor_pairs)
    if pred_count == tgt_count and pred_count > 0:
        pred_ordered = pred_loop.ordered_anchor_pairs()
        tgt_ordered = tgt_loop.ordered_anchor_pairs()
        if loop_type == LoopType.MULTILOOP and pred_count > 2:
            return _cyclic_anchor_pairs_score(pred_ordered, tgt_ordered)
        return anchor_pairs_score(pred_ordered, tgt_ordered)
    return anchor_pairs_score(pred_loop.anchor_pairs, tgt_loop.anchor_pairs)


def loop_anchor_pairs_score_any(pred_loop, tgt_loop) -> float:
    pred_count = len(pred_loop.anchor_pairs)
    tgt_count = len(tgt_loop.anchor_pairs)
    if pred_count != tgt_count:
        return 0.0
    if pred_count == 0:
        return 1.0
    pred_ordered = pred_loop.ordered_anchor_pairs()
    tgt_ordered = tgt_loop.ordered_anchor_pairs()
    if pred_count > 2:
        return _cyclic_anchor_pairs_score(pred_ordered, tgt_ordered)
    return anchor_pairs_score(pred_ordered, tgt_ordered)


def _cyclic_anchor_pairs_score(pred_anchors, tgt_anchors) -> float:
    pred_norm = [tuple(sorted(pair)) for pair in pred_anchors]
    tgt_norm = [tuple(sorted(pair)) for pair in tgt_anchors]
    count = len(pred_norm)
    if count == 0:
        return 1.0
    for shift in range(count):
        if all(pred_norm[idx] == tgt_norm[(idx + shift) % count] for idx in range(count)):
            return 1.0
    return 0.0


def _ordered_anchor_ids_score(pred_ids: list[int], tgt_ids: list[int]) -> float:
    if not pred_ids and not tgt_ids:
        return 1.0
    if not pred_ids or not tgt_ids:
        return 0.0
    if len(pred_ids) != len(tgt_ids):
        return 0.0
    matches = sum(1 for pred_id, tgt_id in zip(pred_ids, tgt_ids) if pred_id >= 0 and pred_id == tgt_id)
    return matches / float(len(pred_ids))


def _cyclic_anchor_ids_score(pred_ids: list[int], tgt_ids: list[int]) -> float:
    count = len(pred_ids)
    if count == 0:
        return 1.0
    best = 0.0
    for shift in range(count):
        matches = 0
        for idx in range(count):
            pred_id = pred_ids[idx]
            tgt_id = tgt_ids[(idx + shift) % count]
            if pred_id >= 0 and pred_id == tgt_id:
                matches += 1
        if matches > best:
            best = matches
    return best / float(count)


def _unordered_anchor_ids_score(pred_ids: list[int], tgt_ids: list[int]) -> float:
    if not pred_ids and not tgt_ids:
        return 1.0
    if not pred_ids or not tgt_ids:
        return 0.0
    pred_counts = Counter(anchor_id for anchor_id in pred_ids if anchor_id >= 0)
    tgt_counts = Counter(anchor_id for anchor_id in tgt_ids if anchor_id >= 0)
    if not pred_counts or not tgt_counts:
        return 0.0
    matches = sum(min(pred_counts[key], tgt_counts.get(key, 0)) for key in pred_counts)
    denom = max(len(pred_ids), len(tgt_ids))
    return matches / float(denom) if denom else 0.0


def _helix_id(helix, helix_key_map: dict[tuple[int, int, int, int], int]) -> int:
    return helix_key_map.get(helix.key, -1)


def _loop_anchor_helix_ids(loop, helix_key_map: dict[tuple[int, int, int, int], int]) -> list[int]:
    anchors = loop.ordered_anchor_helices()
    if not anchors:
        return []
    return [_helix_id(helix, helix_key_map) for helix in anchors]


def _map_anchor_ids(pred_ids: list[int], pred_to_tgt: dict[int, int]) -> list[int]:
    return [pred_to_tgt.get(anchor_id, -1) if anchor_id >= 0 else -1 for anchor_id in pred_ids]


def loop_anchor_wiring_score(
    loop_type: LoopType,
    pred_loop,
    tgt_loop,
    pred_helix_key_map: dict[tuple[int, int, int, int], int],
    tgt_helix_key_map: dict[tuple[int, int, int, int], int],
    pred_to_tgt: dict[int, int],
) -> float:
    pred_ids = _loop_anchor_helix_ids(pred_loop, pred_helix_key_map)
    tgt_ids = _loop_anchor_helix_ids(tgt_loop, tgt_helix_key_map)
    if not pred_ids and not tgt_ids:
        return 1.0
    if not pred_ids or not tgt_ids:
        return 0.0
    mapped_ids = _map_anchor_ids(pred_ids, pred_to_tgt)
    if len(mapped_ids) == len(tgt_ids):
        if loop_type == LoopType.MULTILOOP and len(mapped_ids) > 2:
            return _cyclic_anchor_ids_score(mapped_ids, tgt_ids)
        return _ordered_anchor_ids_score(mapped_ids, tgt_ids)
    return _unordered_anchor_ids_score(mapped_ids, tgt_ids)


def _loop_anchor_adjacency_edges(
    loop, helix_key_map: dict[tuple[int, int, int, int], int], helix_edges: list[tuple[int, int, int]]
) -> set[tuple[int, int, int]]:
    anchor_ids = {idx for idx in _loop_anchor_helix_ids(loop, helix_key_map) if idx >= 0}
    if not anchor_ids:
        return set()
    return {(src, dst, edge_type) for src, dst, edge_type in helix_edges if src in anchor_ids and dst in anchor_ids}


def _loop_anchor_order_edges(loop, helix_key_map: dict[tuple[int, int, int, int], int]) -> list[tuple[int, int]]:
    anchor_ids = _loop_anchor_helix_ids(loop, helix_key_map)
    ordered = [anchor_id for anchor_id in anchor_ids if anchor_id >= 0]
    if len(ordered) <= 2:
        return []
    edges: list[tuple[int, int]] = []
    count = len(ordered)
    for idx in range(count):
        edges.append((ordered[idx], ordered[(idx + 1) % count]))
    return edges


def loop_anchor_adjacency_score(
    pred_loop,
    tgt_loop,
    pred_helix_key_map: dict[tuple[int, int, int, int], int],
    tgt_helix_key_map: dict[tuple[int, int, int, int], int],
    pred_to_tgt: dict[int, int],
    pred_helix_edges: list[tuple[int, int, int]],
    tgt_helix_edges: list[tuple[int, int, int]],
) -> float:
    pred_edges = _loop_anchor_adjacency_edges(pred_loop, pred_helix_key_map, pred_helix_edges)
    tgt_edges = _loop_anchor_adjacency_edges(tgt_loop, tgt_helix_key_map, tgt_helix_edges)
    if not pred_edges and not tgt_edges:
        return 1.0
    if not pred_edges or not tgt_edges:
        return 0.0
    mapped = {(pred_to_tgt.get(src, -1), pred_to_tgt.get(dst, -1), edge_type) for src, dst, edge_type in pred_edges}
    mapped = {edge for edge in mapped if edge[0] >= 0 and edge[1] >= 0}
    if not mapped:
        return 0.0
    inter = len(mapped & tgt_edges)
    union = len(mapped | tgt_edges)
    return (inter / union) if union > 0 else 0.0


def _loop_anchor_order_score(
    loop_type: LoopType,
    pred_loop,
    tgt_loop,
    pred_helix_key_map: dict[tuple[int, int, int, int], int],
    tgt_helix_key_map: dict[tuple[int, int, int, int], int],
    pred_to_tgt: dict[int, int],
) -> float:
    if loop_type != LoopType.MULTILOOP:
        return 1.0
    pred_edges = _loop_anchor_order_edges(pred_loop, pred_helix_key_map)
    tgt_edges = _loop_anchor_order_edges(tgt_loop, tgt_helix_key_map)
    if not pred_edges and not tgt_edges:
        return 1.0
    if not pred_edges or not tgt_edges:
        return 0.0
    mapped = {(pred_to_tgt.get(src, -1), pred_to_tgt.get(dst, -1)) for src, dst in pred_edges}
    mapped = {edge for edge in mapped if edge[0] >= 0 and edge[1] >= 0}
    if not mapped:
        return 0.0
    inter = len(mapped & set(tgt_edges))
    union = len(mapped | set(tgt_edges))
    return (inter / union) if union > 0 else 0.0


def loop_anchor_order_score_any(
    pred_loop,
    tgt_loop,
    pred_helix_key_map: dict[tuple[int, int, int, int], int],
    tgt_helix_key_map: dict[tuple[int, int, int, int], int],
    pred_to_tgt: dict[int, int],
) -> float:
    pred_edges = _loop_anchor_order_edges(pred_loop, pred_helix_key_map)
    tgt_edges = _loop_anchor_order_edges(tgt_loop, tgt_helix_key_map)
    if not pred_edges and not tgt_edges:
        return 1.0
    if not pred_edges or not tgt_edges:
        return 0.0
    mapped = {(pred_to_tgt.get(src, -1), pred_to_tgt.get(dst, -1)) for src, dst in pred_edges}
    mapped = {edge for edge in mapped if edge[0] >= 0 and edge[1] >= 0}
    if not mapped:
        return 0.0
    inter = len(mapped & set(tgt_edges))
    union = len(mapped | set(tgt_edges))
    return (inter / union) if union > 0 else 0.0


def _loop_anchor_coaxial_edges(
    loop, helix_key_map: dict[tuple[int, int, int, int], int], helix_edges: list[tuple[int, int, int]]
) -> set[tuple[int, int, int]]:
    anchor_ids = _loop_anchor_helix_ids(loop, helix_key_map)
    ordered = [anchor_id for anchor_id in anchor_ids if anchor_id >= 0]
    if len(ordered) < 2:
        return set()
    if len(ordered) == 2:
        order_pairs = {(ordered[0], ordered[1]), (ordered[1], ordered[0])}
    else:
        order_edges = _loop_anchor_order_edges(loop, helix_key_map)
        order_pairs = set(order_edges) | {(dst, src) for src, dst in order_edges}
    anchor_set = set(ordered)
    return {
        (src, dst, edge_type)
        for src, dst, edge_type in helix_edges
        if src in anchor_set and dst in anchor_set and (src, dst) in order_pairs
    }


def loop_anchor_coaxial_score(
    pred_loop,
    tgt_loop,
    pred_helix_key_map: dict[tuple[int, int, int, int], int],
    tgt_helix_key_map: dict[tuple[int, int, int, int], int],
    pred_to_tgt: dict[int, int],
    pred_helix_edges: list[tuple[int, int, int]],
    tgt_helix_edges: list[tuple[int, int, int]],
) -> float:
    pred_edges = _loop_anchor_coaxial_edges(pred_loop, pred_helix_key_map, pred_helix_edges)
    tgt_edges = _loop_anchor_coaxial_edges(tgt_loop, tgt_helix_key_map, tgt_helix_edges)
    if not pred_edges and not tgt_edges:
        return 1.0
    if not pred_edges or not tgt_edges:
        return 0.0
    mapped = {(pred_to_tgt.get(src, -1), pred_to_tgt.get(dst, -1), edge_type) for src, dst, edge_type in pred_edges}
    mapped = {edge for edge in mapped if edge[0] >= 0 and edge[1] >= 0}
    if not mapped:
        return 0.0
    inter = len(mapped & tgt_edges)
    union = len(mapped | tgt_edges)
    return (inter / union) if union > 0 else 0.0


def _loop_confusion_matrix(loop_type: LoopType, context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_loops, target_loops = context.loops_by_type[loop_type]
    pred_total = len(pred_loops)
    target_total = len(target_loops)
    if target_total == 0:
        nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
        return make_confusion_matrix(nan, nan, nan, nan)
    if pred_total == 0:
        target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
        zero = target_total_t.new_zeros(())
        return make_confusion_matrix(zero, zero, target_total_t)

    length = getattr(context, "length", None)
    if length is None:
        max_stop = -1
        for loop in (*pred_loops, *target_loops):
            for span in loop.spans:
                max_stop = max(max_stop, int(span.stop))
        length = max_stop + 1 if max_stop >= 0 else 0

    pred_sizes, target_sizes, overlap, valid = loop_overlap_components(
        pred_loops,
        target_loops,
        length=length,
        device=device,
    )
    score_mat = torch.full((pred_total, target_total), -1.0, device=device)
    if valid.any():
        pred_sizes_f = pred_sizes[:, None]
        target_sizes_f = target_sizes[None, :]
        union_len = pred_sizes_f + target_sizes_f - overlap
        jaccard = torch.where(overlap > 0, overlap / union_len, overlap.new_zeros(()))
        score = jaccard + overlap * MATCH_TIEBREAK_OVERLAP + target_sizes_f * MATCH_TIEBREAK_SIZE
        score_mat[valid] = score[valid]

    matched_pred, _ = bipartite_match(score_mat)
    tp = torch.tensor(matched_pred.numel(), device=device, dtype=torch.float32)
    pred_total_t = torch.tensor(pred_total, device=device, dtype=torch.float32)
    target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
    fp = pred_total_t - tp
    fn = target_total_t - tp
    return make_confusion_matrix(tp, fp, fn)


def _loop_nucleotide_labels(loop_type: LoopType, context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor]:
    return context.loop_nt_labels(loop_type)


def _segment_confusion_matrix(segment_type: LoopSegmentType, context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_segments, target_segments = context.loop_segments_by_kind[segment_type]
    pred_total = len(pred_segments)
    target_total = len(target_segments)
    if target_total == 0:
        nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
        return make_confusion_matrix(nan, nan, nan, nan)
    if pred_total == 0:
        target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
        zero = target_total_t.new_zeros(())
        return make_confusion_matrix(zero, zero, target_total_t)

    pred_len, tgt_len, overlap, valid = segment_overlap_components(
        pred_segments,
        target_segments,
        device=device,
        enforce_end_side=segment_type == LoopSegmentType.END,
    )

    score_mat = torch.full((pred_total, target_total), -1.0, device=device)
    if valid.any():
        pred_len_f = pred_len[:, None]
        tgt_len_f = tgt_len[None, :]
        union_len = pred_len_f + tgt_len_f - overlap
        jaccard = torch.where(overlap > 0, overlap / union_len, overlap.new_zeros(()))
        score = jaccard + overlap * MATCH_TIEBREAK_OVERLAP + tgt_len_f * MATCH_TIEBREAK_SIZE
        score_mat[valid] = score[valid]

    matched_pred, _ = bipartite_match(score_mat)
    tp = torch.tensor(matched_pred.numel(), device=device, dtype=torch.float32)
    pred_total_t = torch.tensor(pred_total, device=device, dtype=torch.float32)
    target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
    fp = pred_total_t - tp
    fn = target_total_t - tp
    return make_confusion_matrix(tp, fp, fn)


# ── Bulge loop overlap metrics ──


def bulge_loops_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _loop_confusion_matrix(LoopType.BULGE, context)


def bulge_loops_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.loop_overlap_ratios(LoopType.BULGE)
    return f1_from_pr(p, r)


def bulge_loops_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.loop_overlap_ratios(LoopType.BULGE)
    return p


def bulge_loops_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.loop_overlap_ratios(LoopType.BULGE)
    return r


# ── Bulge nucleotide metrics ──


def bulge_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.BULGE, context)
    return binary_confusion_matrix(preds, targets)


def bulge_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.BULGE, context)
    return binary_f1_score(preds, targets)


def bulge_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.BULGE, context)
    return binary_mcc(preds, targets)


def bulge_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.BULGE, context)
    return binary_precision(preds, targets)


def bulge_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.BULGE, context)
    return binary_recall(preds, targets)


# ── External loop overlap metrics ──


def external_loops_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _loop_confusion_matrix(LoopType.EXTERNAL, context)


def external_loops_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.loop_overlap_ratios(LoopType.EXTERNAL)
    return f1_from_pr(p, r)


def external_loops_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.loop_overlap_ratios(LoopType.EXTERNAL)
    return p


def external_loops_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.loop_overlap_ratios(LoopType.EXTERNAL)
    return r


# ── External nucleotide metrics ──


def external_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.EXTERNAL, context)
    return binary_confusion_matrix(preds, targets)


def external_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.EXTERNAL, context)
    return binary_f1_score(preds, targets)


def external_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.EXTERNAL, context)
    return binary_mcc(preds, targets)


def external_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.EXTERNAL, context)
    return binary_precision(preds, targets)


def external_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.EXTERNAL, context)
    return binary_recall(preds, targets)


# ── Hairpin loop overlap metrics ──


def hairpin_loops_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _loop_confusion_matrix(LoopType.HAIRPIN, context)


def hairpin_loops_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.loop_overlap_ratios(LoopType.HAIRPIN)
    return f1_from_pr(p, r)


def hairpin_loops_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.loop_overlap_ratios(LoopType.HAIRPIN)
    return p


def hairpin_loops_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.loop_overlap_ratios(LoopType.HAIRPIN)
    return r


# ── Hairpin nucleotide metrics ──


def hairpin_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.HAIRPIN, context)
    return binary_confusion_matrix(preds, targets)


def hairpin_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.HAIRPIN, context)
    return binary_f1_score(preds, targets)


def hairpin_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.HAIRPIN, context)
    return binary_mcc(preds, targets)


def hairpin_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.HAIRPIN, context)
    return binary_precision(preds, targets)


def hairpin_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.HAIRPIN, context)
    return binary_recall(preds, targets)


# ── Internal loop overlap metrics ──


def internal_loops_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _loop_confusion_matrix(LoopType.INTERNAL, context)


def internal_loops_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.loop_overlap_ratios(LoopType.INTERNAL)
    return f1_from_pr(p, r)


def internal_loops_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.loop_overlap_ratios(LoopType.INTERNAL)
    return p


def internal_loops_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.loop_overlap_ratios(LoopType.INTERNAL)
    return r


# ── Internal nucleotide metrics ──


def internal_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.INTERNAL, context)
    return binary_confusion_matrix(preds, targets)


def internal_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.INTERNAL, context)
    return binary_f1_score(preds, targets)


def internal_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.INTERNAL, context)
    return binary_mcc(preds, targets)


def internal_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.INTERNAL, context)
    return binary_precision(preds, targets)


def internal_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.INTERNAL, context)
    return binary_recall(preds, targets)


# ── Multiloop loop overlap metrics ──


def multiloop_loops_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _loop_confusion_matrix(LoopType.MULTILOOP, context)


def multiloop_loops_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.loop_overlap_ratios(LoopType.MULTILOOP)
    return f1_from_pr(p, r)


def multiloop_loops_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.loop_overlap_ratios(LoopType.MULTILOOP)
    return p


def multiloop_loops_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.loop_overlap_ratios(LoopType.MULTILOOP)
    return r


# ── Multiloop nucleotide metrics ──


def multiloop_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.MULTILOOP, context)
    return binary_confusion_matrix(preds, targets)


def multiloop_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.MULTILOOP, context)
    return binary_f1_score(preds, targets)


def multiloop_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.MULTILOOP, context)
    return binary_mcc(preds, targets)


def multiloop_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.MULTILOOP, context)
    return binary_precision(preds, targets)


def multiloop_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _loop_nucleotide_labels(LoopType.MULTILOOP, context)
    return binary_recall(preds, targets)


# ── Hairpin segment metrics ──


def hairpin_segments_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _segment_confusion_matrix(LoopSegmentType.HAIRPIN, context)


def hairpin_segments_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.segment_overlap_ratios(LoopSegmentType.HAIRPIN)
    return f1_from_pr(p, r)


def hairpin_segments_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.segment_overlap_ratios(LoopSegmentType.HAIRPIN)
    return p


def hairpin_segments_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.segment_overlap_ratios(LoopSegmentType.HAIRPIN)
    return r


# ── Bulge segment metrics ──


def bulge_segments_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _segment_confusion_matrix(LoopSegmentType.BULGE, context)


def bulge_segments_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.segment_overlap_ratios(LoopSegmentType.BULGE)
    return f1_from_pr(p, r)


def bulge_segments_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.segment_overlap_ratios(LoopSegmentType.BULGE)
    return p


def bulge_segments_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.segment_overlap_ratios(LoopSegmentType.BULGE)
    return r


# ── Internal segment metrics ──


def internal_segments_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _segment_confusion_matrix(LoopSegmentType.INTERNAL, context)


def internal_segments_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.segment_overlap_ratios(LoopSegmentType.INTERNAL)
    return f1_from_pr(p, r)


def internal_segments_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.segment_overlap_ratios(LoopSegmentType.INTERNAL)
    return p


def internal_segments_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.segment_overlap_ratios(LoopSegmentType.INTERNAL)
    return r


# ── Multiloop segment metrics ──


def multiloop_segments_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _segment_confusion_matrix(LoopSegmentType.BRANCH, context)


def multiloop_segments_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.segment_overlap_ratios(LoopSegmentType.BRANCH)
    return f1_from_pr(p, r)


def multiloop_segments_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.segment_overlap_ratios(LoopSegmentType.BRANCH)
    return p


def multiloop_segments_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.segment_overlap_ratios(LoopSegmentType.BRANCH)
    return r


# ── External segment metrics ──


def external_segments_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _segment_confusion_matrix(LoopSegmentType.EXTERNAL, context)


def external_segments_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.segment_overlap_ratios(LoopSegmentType.EXTERNAL)
    return f1_from_pr(p, r)


def external_segments_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.segment_overlap_ratios(LoopSegmentType.EXTERNAL)
    return p


def external_segments_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.segment_overlap_ratios(LoopSegmentType.EXTERNAL)
    return r


# ── End segment metrics ──


def end_segments_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    return _segment_confusion_matrix(LoopSegmentType.END, context)


def end_segments_f1(context: RnaSecondaryStructureContext) -> Tensor:
    p, r = context.segment_overlap_ratios(LoopSegmentType.END)
    return f1_from_pr(p, r)


def end_segments_precision(context: RnaSecondaryStructureContext) -> Tensor:
    p, _ = context.segment_overlap_ratios(LoopSegmentType.END)
    return p


def end_segments_recall(context: RnaSecondaryStructureContext) -> Tensor:
    _, r = context.segment_overlap_ratios(LoopSegmentType.END)
    return r
