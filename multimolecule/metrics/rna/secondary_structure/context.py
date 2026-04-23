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

from functools import cached_property

import torch
from torch import Tensor

from ....utils.rna.secondary_structure import (
    LoopSegmentType,
    LoopType,
    RnaSecondaryStructureTopology,
    StructureView,
    contact_map_to_pairs,
)
from .common import (
    MATCH_TIEBREAK_OVERLAP,
    MATCH_TIEBREAK_SIZE,
    approximate_graph_edit_distance,
    bipartite_match,
    helix_score_matrix,
    loop_overlap_components,
    loop_substitution_cost,
    make_confusion_matrix,
    segment_length_cost,
    segment_overlap_components,
    segment_type_from_loop,
    segments_from_topology,
)
from .stems import (
    stem_confusion_from_arrays,
    stem_ids_from_arrays,
    stem_ids_from_events,
    stem_match_indices,
    stem_segment_data,
)


class RnaSecondaryStructureContext:
    sequence: str
    length: int
    threshold: float
    pred: Tensor
    pred_pairs: Tensor
    target_pairs: Tensor
    pred_topology: RnaSecondaryStructureTopology
    target_topology: RnaSecondaryStructureTopology
    pred_noncanonical_pairs: Tensor
    target_noncanonical_pairs: Tensor
    _loop_nt_cache: dict
    _segment_nt_cache: dict
    _loop_overlap_cache: dict
    _segment_overlap_cache: dict

    def __init__(
        self,
        pred: Tensor,
        target: Tensor,
        sequence: str,
        *,
        threshold: float = 0.5,
    ) -> None:
        if pred.ndim != 2 or pred.shape[0] != pred.shape[1]:
            raise ValueError("pred must be a square 2D contact map")
        if target.ndim != 2 or target.shape[0] != target.shape[1]:
            raise ValueError("target must be a square 2D contact map")
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")
        length = int(pred.shape[0])
        if len(sequence) != length:
            raise ValueError("sequence length must match contact map size")
        self.sequence = sequence
        self.length = length
        self.threshold = float(threshold)
        self.pred = pred

        pred_pairs = contact_map_to_pairs(pred, unsafe=True, threshold=threshold)
        target_pairs = contact_map_to_pairs(target, unsafe=True, threshold=threshold)
        self.pred_pairs = pred_pairs
        self.target_pairs = target_pairs

        pred_topology = RnaSecondaryStructureTopology(sequence, pred_pairs)
        target_topology = RnaSecondaryStructureTopology(sequence, target_pairs)

        self.pred_topology = pred_topology
        self.target_topology = target_topology
        self.pred_noncanonical_pairs = pred_topology.noncanonical_pairs(unsafe=True)
        self.target_noncanonical_pairs = target_topology.noncanonical_pairs(unsafe=True)

        self.device = pred.device
        self._loop_nt_cache = {}
        self._segment_nt_cache = {}
        self._loop_overlap_cache = {}
        self._segment_overlap_cache = {}

    @cached_property
    def _stem_data(self) -> tuple:
        base = self.length + 1
        pred = stem_segment_data(self.pred_pairs, base)
        target = stem_segment_data(self.target_pairs, base)
        return pred, target, base

    @cached_property
    def pred_stem_start_i(self) -> Tensor:
        return self._stem_data[0][0]

    @cached_property
    def pred_stem_start_j(self) -> Tensor:
        return self._stem_data[0][1]

    @cached_property
    def pred_stem_lengths(self) -> Tensor:
        return self._stem_data[0][2]

    @cached_property
    def pred_stem_pair_ids(self) -> Tensor:
        return self._stem_data[0][3]

    @cached_property
    def pred_stem_pair_stem(self) -> Tensor:
        return self._stem_data[0][4]

    @cached_property
    def target_stem_start_i(self) -> Tensor:
        return self._stem_data[1][0]

    @cached_property
    def target_stem_start_j(self) -> Tensor:
        return self._stem_data[1][1]

    @cached_property
    def target_stem_lengths(self) -> Tensor:
        return self._stem_data[1][2]

    @cached_property
    def target_stem_pair_ids(self) -> Tensor:
        return self._stem_data[1][3]

    @cached_property
    def target_stem_pair_stem(self) -> Tensor:
        return self._stem_data[1][4]

    @cached_property
    def pred_stem_ids(self) -> Tensor:
        base = self._stem_data[2]
        return stem_ids_from_arrays(self.pred_stem_start_i, self.pred_stem_start_j, self.pred_stem_lengths, base)

    @cached_property
    def target_stem_ids(self) -> Tensor:
        base = self._stem_data[2]
        return stem_ids_from_arrays(self.target_stem_start_i, self.target_stem_start_j, self.target_stem_lengths, base)

    @cached_property
    def _crossing_data(self) -> tuple:
        base = self.length + 1
        pred_events = self.pred_topology.crossing_events
        target_events = self.target_topology.crossing_events
        pred_ce = (
            pred_events.new_empty((0, 8), dtype=torch.long)
            if pred_events.numel() == 0
            else pred_events.reshape(-1, 8).to(dtype=torch.long)
        )
        target_ce = (
            target_events.new_empty((0, 8), dtype=torch.long)
            if target_events.numel() == 0
            else target_events.reshape(-1, 8).to(dtype=torch.long)
        )
        pred_cs = stem_ids_from_events(pred_events, base)
        target_cs = stem_ids_from_events(target_events, base)
        return pred_ce, target_ce, pred_cs, target_cs

    @cached_property
    def pred_crossing_events(self) -> Tensor:
        return self._crossing_data[0]

    @cached_property
    def target_crossing_events(self) -> Tensor:
        return self._crossing_data[1]

    @cached_property
    def pred_crossing_stem_ids(self) -> Tensor:
        return self._crossing_data[2]

    @cached_property
    def target_crossing_stem_ids(self) -> Tensor:
        return self._crossing_data[3]

    @cached_property
    def loop_segments(self) -> tuple[list, list]:
        return segments_from_topology(self.pred_topology), segments_from_topology(self.target_topology)

    @cached_property
    def loop_segments_by_kind(self) -> dict[LoopSegmentType, tuple[list, list]]:
        pred_segments, tgt_segments = self.loop_segments
        pred_by_kind: dict[LoopSegmentType, list] = {kind: [] for kind in LoopSegmentType}
        tgt_by_kind: dict[LoopSegmentType, list] = {kind: [] for kind in LoopSegmentType}
        for segment in pred_segments:
            pred_by_kind[segment.kind].append(segment)
        for segment in tgt_segments:
            tgt_by_kind[segment.kind].append(segment)
        return {kind: (pred_by_kind[kind], tgt_by_kind[kind]) for kind in pred_by_kind}

    @cached_property
    def loops_by_type(self) -> dict[LoopType, tuple[list, list]]:
        pred_loops = self.pred_topology.taxonomy_loops
        tgt_loops = self.target_topology.taxonomy_loops
        return {
            LoopType.HAIRPIN: (pred_loops.hairpin_loops, tgt_loops.hairpin_loops),
            LoopType.BULGE: (pred_loops.bulge_loops, tgt_loops.bulge_loops),
            LoopType.INTERNAL: (pred_loops.internal_loops, tgt_loops.internal_loops),
            LoopType.MULTILOOP: (pred_loops.multi_loops, tgt_loops.multi_loops),
            LoopType.EXTERNAL: (pred_loops.external_loops, tgt_loops.external_loops),
        }

    @cached_property
    def paired_labels(self) -> tuple[Tensor, Tensor]:
        length = self.length
        device = self.device
        pred = torch.zeros(length, dtype=torch.int64, device=device)
        target = torch.zeros(length, dtype=torch.int64, device=device)
        if self.pred_pairs.numel():
            idx = self.pred_pairs.to(dtype=torch.long).view(-1)
            pred.index_fill_(0, idx, 1)
        if self.target_pairs.numel():
            idx = self.target_pairs.to(dtype=torch.long).view(-1)
            target.index_fill_(0, idx, 1)
        return pred, target

    def loop_nt_labels(self, loop_type: LoopType) -> tuple[Tensor, Tensor]:
        cached = self._loop_nt_cache.get(loop_type)
        if cached is not None:
            return cached
        segment_type = segment_type_from_loop(loop_type)
        labels = self.segment_nt_labels(segment_type)
        self._loop_nt_cache[loop_type] = labels
        return labels

    def segment_nt_labels(self, segment_type: LoopSegmentType) -> tuple[Tensor, Tensor]:
        cached = self._segment_nt_cache.get(segment_type)
        if cached is not None:
            return cached
        length = self.length
        device = self.device
        pred_segments, tgt_segments = self.loop_segments
        pred_masks = self._segment_nucleotide_masks(pred_segments, length, device)
        target_masks = self._segment_nucleotide_masks(tgt_segments, length, device)
        for kind in LoopSegmentType:
            self._segment_nt_cache[kind] = (pred_masks[kind], target_masks[kind])
        return self._segment_nt_cache[segment_type]

    @cached_property
    def helix_segments(self) -> tuple[list, list]:
        pred_helices = self.pred_topology.helices(view=StructureView.ALL)
        tgt_helices = self.target_topology.helices(view=StructureView.ALL)
        return pred_helices, tgt_helices

    @cached_property
    def helix_key_maps(self) -> tuple[dict, dict]:
        pred_helices, tgt_helices = self.helix_segments
        return (
            {helix.key: idx for idx, helix in enumerate(pred_helices)},
            {helix.key: idx for idx, helix in enumerate(tgt_helices)},
        )

    @cached_property
    def helix_assignment(self) -> tuple[dict[int, int], dict[int, int]]:
        pred_helices, tgt_helices = self.helix_segments
        score_mat = helix_score_matrix(pred_helices, tgt_helices, self.device)
        matched_pred, matched_tgt = bipartite_match(score_mat)
        pred_to_tgt = {
            int(pred_idx): int(tgt_idx) for pred_idx, tgt_idx in zip(matched_pred.tolist(), matched_tgt.tolist())
        }
        tgt_to_pred = {tgt_idx: pred_idx for pred_idx, tgt_idx in pred_to_tgt.items()}
        return pred_to_tgt, tgt_to_pred

    @cached_property
    def helix_edge_lists(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        pred_key_map, tgt_key_map = self.helix_key_maps
        return (
            self._helix_edges_from_topology(self.pred_topology, pred_key_map),
            self._helix_edges_from_topology(self.target_topology, tgt_key_map),
        )

    @cached_property
    def loop_helix_components(self) -> tuple:
        pred_helices, tgt_helices = self.helix_segments
        pred_helix_key_map, tgt_helix_key_map = self.helix_key_maps
        pred_loops, pred_helices, pred_edges = self._loop_helix_components_from_topology(
            self.pred_topology,
            pred_helices,
            pred_helix_key_map,
        )
        tgt_loops, tgt_helices, tgt_edges = self._loop_helix_components_from_topology(
            self.target_topology,
            tgt_helices,
            tgt_helix_key_map,
        )
        pred_edges_unique = sorted(set(pred_edges))
        tgt_edges_unique = sorted(set(tgt_edges))
        pred_loop_adj, pred_helix_adj = self._adjacency_lists(
            pred_edges_unique,
            len(pred_loops),
            len(pred_helices),
        )
        tgt_loop_adj, tgt_helix_adj = self._adjacency_lists(
            tgt_edges_unique,
            len(tgt_loops),
            len(tgt_helices),
        )
        return (
            pred_loops,
            tgt_loops,
            pred_helices,
            tgt_helices,
            pred_edges_unique,
            tgt_edges_unique,
            pred_loop_adj,
            pred_helix_adj,
            tgt_loop_adj,
            tgt_helix_adj,
        )

    @cached_property
    def loop_helix_assignment(self) -> tuple:
        from .adjacency import _global_node_assignment, _joint_node_score, _loop_score_matrix

        device = self.device
        (
            pred_loops,
            tgt_loops,
            pred_helices,
            tgt_helices,
            pred_edges_unique,
            tgt_edges_unique,
            pred_loop_adj,
            pred_helix_adj,
            tgt_loop_adj,
            tgt_helix_adj,
        ) = self.loop_helix_components
        pred_helix_key_map, tgt_helix_key_map = self.helix_key_maps
        pred_to_tgt, _ = self.helix_assignment
        pred_helix_edges, tgt_helix_edges = self.helix_edge_lists
        loop_score = _loop_score_matrix(
            pred_loops,
            tgt_loops,
            pred_helix_key_map,
            tgt_helix_key_map,
            pred_to_tgt,
            pred_helix_edges,
            tgt_helix_edges,
            device,
        )
        helix_score = helix_score_matrix(pred_helices, tgt_helices, device)
        loop_score_joint = _joint_node_score(loop_score, pred_loop_adj, tgt_loop_adj, helix_score, device)
        helix_score_joint = _joint_node_score(helix_score, pred_helix_adj, tgt_helix_adj, loop_score, device)
        loop_map, helix_map = _global_node_assignment(loop_score_joint, helix_score_joint, device)
        return (
            pred_loops,
            tgt_loops,
            pred_helices,
            tgt_helices,
            pred_edges_unique,
            tgt_edges_unique,
            loop_map,
            helix_map,
        )

    def loop_overlap_ratios(self, loop_type: LoopType) -> tuple[Tensor, Tensor]:
        return self._overlap_ratios_impl(
            self._loop_overlap_cache,
            loop_type,
            self.loops_by_type[loop_type],
            lambda pred, tgt, device: loop_overlap_components(pred, tgt, length=self.length, device=device),
            use_tiebreaker=True,
        )

    def segment_overlap_ratios(self, segment_type: LoopSegmentType) -> tuple[Tensor, Tensor]:
        return self._overlap_ratios_impl(
            self._segment_overlap_cache,
            segment_type,
            self.loop_segments_by_kind[segment_type],
            lambda pred, tgt, device: segment_overlap_components(
                pred, tgt, device=device, enforce_end_side=segment_type == LoopSegmentType.END
            ),
            use_tiebreaker=False,
        )

    def _overlap_ratios_impl(
        self,
        cache: dict,
        key,
        pred_target_pair: tuple[list, list],
        overlap_fn,
        *,
        use_tiebreaker: bool,
    ) -> tuple[Tensor, Tensor]:
        cached = cache.get(key)
        if cached is not None:
            return cached

        device = self.device
        pred_items, target_items = pred_target_pair
        pred_total = len(pred_items)
        target_total = len(target_items)
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if target_total == 0:
            nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
            res = (nan, nan)
            cache[key] = res
            return res
        if pred_total == 0:
            res = (zero, zero)
            cache[key] = res
            return res
        pred_sizes, target_sizes, overlap, valid = overlap_fn(pred_items, target_items, device)
        if not valid.any():
            res = (zero, zero)
            cache[key] = res
            return res
        score_mat = torch.full((pred_total, target_total), -1.0, device=device)
        overlap_mat = torch.full((pred_total, target_total), -1.0, device=device, dtype=torch.float32)
        pred_sizes_f = pred_sizes[:, None]
        target_sizes_f = target_sizes[None, :]
        union_len = pred_sizes_f + target_sizes_f - overlap
        jaccard = torch.where(overlap > 0, overlap / union_len, overlap.new_zeros(()))
        if use_tiebreaker:
            score = jaccard + overlap * MATCH_TIEBREAK_OVERLAP + target_sizes_f * MATCH_TIEBREAK_SIZE
        else:
            score = jaccard
        score_mat[valid] = score[valid]
        overlap_mat[valid] = overlap[valid]

        matched_pred, matched_tgt = bipartite_match(score_mat)
        if matched_pred.numel() == 0:
            res = (zero, zero)
            cache[key] = res
            return res
        overlaps = overlap_mat[matched_pred, matched_tgt]
        precision_sum = (overlaps / pred_sizes[matched_pred]).sum()
        recall_sum = (overlaps / target_sizes[matched_tgt]).sum()
        pred_total_t = torch.tensor(pred_total, device=device, dtype=torch.float32)
        target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
        precision = precision_sum / pred_total_t
        recall = recall_sum / target_total_t
        res = (precision, recall)
        cache[key] = res
        return res

    @cached_property
    def topology_confusion(self) -> Tensor:
        from .adjacency import _count_consistent_edge_matches
        from .topology import _stem_assignment_and_edges

        device = self.device
        (
            pred_loops,
            tgt_loops,
            pred_helices,
            tgt_helices,
            pred_loop_edges,
            tgt_loop_edges,
            loop_map,
            helix_map,
        ) = self.loop_helix_assignment

        pred_helix_edges, tgt_helix_edges = self.helix_edge_lists
        pred_helix_edge_set = set(pred_helix_edges)
        tgt_helix_edge_set = set(tgt_helix_edges)

        loop_edge_matches = _count_consistent_edge_matches(pred_loop_edges, tgt_loop_edges, loop_map, helix_map)
        helix_edge_matches = 0
        for src, dst, edge_type in pred_helix_edge_set:
            tgt_src = helix_map.get(src)
            tgt_dst = helix_map.get(dst)
            if tgt_src is None or tgt_dst is None:
                continue
            if (tgt_src, tgt_dst, edge_type) in tgt_helix_edge_set:
                helix_edge_matches += 1

        stem_matched, stem_pred_nodes, stem_tgt_nodes, stem_edge_matches, stem_pred_edges, stem_tgt_edges = (
            _stem_assignment_and_edges(self)
        )

        pred_nodes = len(pred_loops) + len(pred_helices) + stem_pred_nodes
        tgt_nodes = len(tgt_loops) + len(tgt_helices) + stem_tgt_nodes
        matched_nodes = len(loop_map) + len(helix_map) + stem_matched

        pred_edges = len(pred_loop_edges) + len(pred_helix_edge_set) + stem_pred_edges
        tgt_edges = len(tgt_loop_edges) + len(tgt_helix_edge_set) + stem_tgt_edges
        matched_edges = loop_edge_matches + helix_edge_matches + stem_edge_matches

        pred_total = pred_nodes + pred_edges
        tgt_total = tgt_nodes + tgt_edges
        if pred_total == 0 and tgt_total == 0:
            return torch.zeros((2, 2), dtype=torch.float32, device=device)

        tp = torch.tensor(float(matched_nodes + matched_edges), device=device)
        pred_total_t = torch.tensor(float(pred_total), device=device)
        tgt_total_t = torch.tensor(float(tgt_total), device=device)
        fp = pred_total_t - tp
        fn = tgt_total_t - tp
        return make_confusion_matrix(tp, fp, fn)

    @cached_property
    def topology_ged(self) -> Tensor:
        device = self.device
        (
            pred_loops,
            tgt_loops,
            pred_helices,
            tgt_helices,
            pred_loop_edges,
            tgt_loop_edges,
            _loop_map,
            _helix_map,
        ) = self.loop_helix_assignment

        pred_helix_edges, tgt_helix_edges = self.helix_edge_lists

        pred_stems, pred_stem_edges_raw = self.pred_topology.stem_graph_components()
        tgt_stems, tgt_stem_edges_raw = self.target_topology.stem_graph_components()
        pred_stem_edges = [(src, dst, ("stem", edge_type)) for src, dst, edge_type in pred_stem_edges_raw]
        tgt_stem_edges = [(src, dst, ("stem", edge_type)) for src, dst, edge_type in tgt_stem_edges_raw]

        pred_nodes = [("loop", loop) for loop in pred_loops] + [("helix", helix) for helix in pred_helices]
        pred_nodes += [("stem", stem) for stem in pred_stems]
        tgt_nodes = [("loop", loop) for loop in tgt_loops] + [("helix", helix) for helix in tgt_helices]
        tgt_nodes += [("stem", stem) for stem in tgt_stems]

        helix_offset_pred = len(pred_loops)
        stem_offset_pred = helix_offset_pred + len(pred_helices)
        helix_offset_tgt = len(tgt_loops)
        stem_offset_tgt = helix_offset_tgt + len(tgt_helices)

        pred_edges: list[tuple[int, int, object]] = []
        tgt_edges: list[tuple[int, int, object]] = []
        for loop_idx, helix_idx in pred_loop_edges:
            pred_edges.append((int(loop_idx), helix_offset_pred + int(helix_idx), ("loop_helix", 0)))
        for loop_idx, helix_idx in tgt_loop_edges:
            tgt_edges.append((int(loop_idx), helix_offset_tgt + int(helix_idx), ("loop_helix", 0)))
        for src, dst, edge_type in pred_helix_edges:
            pred_edges.append((helix_offset_pred + int(src), helix_offset_pred + int(dst), ("helix", int(edge_type))))
        for src, dst, edge_type in tgt_helix_edges:
            tgt_edges.append((helix_offset_tgt + int(src), helix_offset_tgt + int(dst), ("helix", int(edge_type))))
        for src, dst, label in pred_stem_edges:
            pred_edges.append((stem_offset_pred + int(src), stem_offset_pred + int(dst), label))
        for src, dst, label in tgt_stem_edges:
            tgt_edges.append((stem_offset_tgt + int(src), stem_offset_tgt + int(dst), label))

        def _node_cost(a, b) -> float:
            type_a, obj_a = a
            type_b, obj_b = b
            if type_a != type_b:
                return 1.0
            if type_a == "loop":
                return loop_substitution_cost(obj_a, obj_b)
            return segment_length_cost(obj_a, obj_b)

        return approximate_graph_edit_distance(
            pred_nodes,
            tgt_nodes,
            pred_edges,
            tgt_edges,
            node_cost_fn=_node_cost,
            device=device,
        )

    @cached_property
    def loop_helix_edges_confusion(self) -> Tensor:
        from .adjacency import _count_consistent_edge_matches

        device = self.device
        (
            _pred_loops,
            _tgt_loops,
            _pred_helices,
            _tgt_helices,
            pred_edges_unique,
            tgt_edges_unique,
            loop_map,
            helix_map,
        ) = self.loop_helix_assignment
        pred_total = len(pred_edges_unique)
        tgt_total = len(tgt_edges_unique)
        if pred_total == 0 and tgt_total == 0:
            return torch.zeros((2, 2), dtype=torch.float32, device=device)
        edge_matches = _count_consistent_edge_matches(
            pred_edges_unique,
            tgt_edges_unique,
            loop_map,
            helix_map,
        )

        tp = torch.tensor(float(edge_matches), device=device)
        pred_total_t = torch.tensor(float(pred_total), device=device)
        tgt_total_t = torch.tensor(float(tgt_total), device=device)
        fp = pred_total_t - tp
        fn = tgt_total_t - tp
        return make_confusion_matrix(tp, fp, fn)

    @staticmethod
    def _helix_edges_from_topology(
        topology: RnaSecondaryStructureTopology,
        helix_key_map: dict[tuple[int, int, int, int], int],
    ) -> list[tuple[int, int, int]]:
        segments, tier_edges = topology.helix_graph_components()
        local_to_global: dict[int, int] = {}
        for idx, helix in enumerate(segments):
            global_idx = helix_key_map.get(helix.key)
            if global_idx is None:
                continue
            local_to_global[int(idx)] = int(global_idx)

        edges: list[tuple[int, int, int]] = []
        for src_local, dst_local, edge_type in tier_edges:
            src = local_to_global.get(int(src_local))
            dst = local_to_global.get(int(dst_local))
            if src is None or dst is None:
                continue
            edges.append((src, dst, int(edge_type)))
        return edges

    @staticmethod
    def _loop_helix_components_from_topology(
        topology: RnaSecondaryStructureTopology,
        helices: list | None,
        helix_key_map: dict[tuple[int, int, int, int], int],
    ) -> tuple:
        loops = topology.taxonomy_loops
        if helices is None:
            helices = topology.helices(view="all")
        edges: list[tuple[int, int]] = []
        for loop_idx, loop in enumerate(loops):
            for helix in loop.anchor_helices:
                helix_idx = helix_key_map.get(helix.key)
                if helix_idx is None:
                    continue
                edges.append((loop_idx, int(helix_idx)))
        return loops, helices, edges

    @staticmethod
    def _adjacency_lists(
        edges: list[tuple[int, int]],
        loop_count: int,
        helix_count: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        loop_adj: list[list[int]] = [[] for _ in range(loop_count)]
        helix_adj: list[list[int]] = [[] for _ in range(helix_count)]
        for loop_idx, helix_idx in edges:
            if 0 <= loop_idx < loop_count and 0 <= helix_idx < helix_count:
                loop_adj[loop_idx].append(helix_idx)
                helix_adj[helix_idx].append(loop_idx)
        return loop_adj, helix_adj

    @cached_property
    def stem_confusion(self) -> Tensor:
        return stem_confusion_from_arrays(
            self.pred_stem_lengths,
            self.pred_stem_pair_ids,
            self.pred_stem_pair_stem,
            self.target_stem_lengths,
            self.target_stem_pair_ids,
            self.target_stem_pair_stem,
        )

    @cached_property
    def crossing_events_confusion(self) -> Tensor:
        from .pseudoknot import _map_event_ids
        from .stems import event_stem_ids, subset_stem_data

        device = self.device
        base = self.length + 1
        pred_events = self.pred_crossing_events
        target_events = self.target_crossing_events
        pred_event_ids = event_stem_ids(pred_events, base)
        target_event_ids = event_stem_ids(target_events, base)
        if pred_event_ids.numel():
            pred_event_ids = torch.unique(pred_event_ids, dim=0)
        if target_event_ids.numel():
            target_event_ids = torch.unique(target_event_ids, dim=0)
        pred_total = int(pred_event_ids.shape[0])
        target_total = int(target_event_ids.shape[0])
        if pred_total == 0 and target_total == 0:
            zero = torch.tensor(0.0, device=device, dtype=torch.float32)
            return make_confusion_matrix(zero, zero, zero)
        if pred_total == 0:
            target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
            zero = target_total_t.new_zeros(())
            return make_confusion_matrix(zero, zero, target_total_t)
        if target_total == 0:
            pred_total_t = torch.tensor(pred_total, device=device, dtype=torch.float32)
            zero = pred_total_t.new_zeros(())
            return make_confusion_matrix(zero, pred_total_t, zero)

        pred_mask = torch.isin(self.pred_stem_ids, self.pred_crossing_stem_ids)
        tgt_mask = torch.isin(self.target_stem_ids, self.target_crossing_stem_ids)
        pred_start_i = self.pred_stem_start_i[pred_mask]
        pred_start_j = self.pred_stem_start_j[pred_mask]
        pred_lengths = self.pred_stem_lengths[pred_mask]
        tgt_start_i = self.target_stem_start_i[tgt_mask]
        tgt_start_j = self.target_stem_start_j[tgt_mask]
        tgt_lengths = self.target_stem_lengths[tgt_mask]
        pred_ids = stem_ids_from_arrays(pred_start_i, pred_start_j, pred_lengths, base)
        tgt_ids = stem_ids_from_arrays(tgt_start_i, tgt_start_j, tgt_lengths, base)
        pred_lengths_sub, pred_pair_ids, pred_pair_stem = subset_stem_data(
            self.pred_stem_lengths,
            self.pred_stem_pair_ids,
            self.pred_stem_pair_stem,
            pred_mask,
        )
        tgt_lengths_sub, tgt_pair_ids, tgt_pair_stem = subset_stem_data(
            self.target_stem_lengths,
            self.target_stem_pair_ids,
            self.target_stem_pair_stem,
            tgt_mask,
        )
        matched_pred, matched_tgt, _ = stem_match_indices(
            pred_lengths_sub,
            pred_pair_ids,
            pred_pair_stem,
            tgt_lengths_sub,
            tgt_pair_ids,
            tgt_pair_stem,
        )
        matched_pred_ids = pred_ids[matched_pred]
        matched_tgt_ids = tgt_ids[matched_tgt]
        mapped = _map_event_ids(pred_event_ids, matched_pred_ids, matched_tgt_ids)
        mapped_mask = (mapped >= 0).all(dim=1)
        mapped = mapped[mapped_mask]
        if mapped.numel() == 0:
            tp = pred_event_ids.new_zeros((), dtype=torch.float32)
        else:
            mapped = torch.sort(mapped, dim=1).values
            target_sorted = torch.sort(target_event_ids, dim=1).values
            mapped = torch.unique(mapped, dim=0)
            target_sorted = torch.unique(target_sorted, dim=0)
            all_events = torch.cat([mapped, target_sorted], dim=0)
            _, counts = torch.unique(all_events, return_counts=True, dim=0)
            tp = (counts == 2).sum().to(dtype=torch.float32)

        pred_total_t = torch.tensor(pred_total, device=device, dtype=torch.float32)
        target_total_t = torch.tensor(target_total, device=device, dtype=torch.float32)
        fp = pred_total_t - tp
        fn = target_total_t - tp
        return make_confusion_matrix(tp, fp, fn)

    @cached_property
    def crossing_stem_confusion(self) -> Tensor:
        from .stems import _stem_arrays_crossing

        pred_lengths, pred_pair_ids, pred_pair_stem, tgt_lengths, tgt_pair_ids, tgt_pair_stem = _stem_arrays_crossing(
            self
        )
        device = self.device
        pred_count = int(pred_lengths.shape[0])
        tgt_count = int(tgt_lengths.shape[0])
        if pred_count == 0 and tgt_count == 0:
            zero = torch.tensor(0.0, device=device, dtype=torch.float32)
            return make_confusion_matrix(zero, zero, zero)
        if pred_count == 0:
            tgt_total = torch.tensor(tgt_count, device=device, dtype=torch.float32)
            zero = tgt_total.new_zeros(())
            return make_confusion_matrix(zero, zero, tgt_total)
        if tgt_count == 0:
            pred_total = torch.tensor(pred_count, device=device, dtype=torch.float32)
            zero = pred_total.new_zeros(())
            return make_confusion_matrix(zero, pred_total, zero)

        matched_pred, _, _ = stem_match_indices(
            pred_lengths,
            pred_pair_ids,
            pred_pair_stem,
            tgt_lengths,
            tgt_pair_ids,
            tgt_pair_stem,
            weight_by_length=False,
        )
        tp = pred_lengths.new_full((), matched_pred.numel(), dtype=torch.float32)
        pred_total = pred_lengths.new_full((), pred_count, dtype=torch.float32)
        tgt_total = pred_lengths.new_full((), tgt_count, dtype=torch.float32)
        fp = pred_total - tp
        fn = tgt_total - tp
        return make_confusion_matrix(tp, fp, fn)

    @cached_property
    def crossing_nt_labels(self) -> tuple[Tensor, Tensor]:
        length = self.length
        device = self.device
        pred_idx = self.pred_topology.crossing_nucleotides
        target_idx = self.target_topology.crossing_nucleotides
        preds = torch.zeros(length, dtype=torch.int64, device=device)
        targets = torch.zeros(length, dtype=torch.int64, device=device)
        if pred_idx.numel():
            preds[pred_idx] = 1
        if target_idx.numel():
            targets[target_idx] = 1
        return preds, targets

    @staticmethod
    def _segment_nucleotide_masks(segments, length: int, device: torch.device) -> dict[LoopSegmentType, Tensor]:
        kinds = tuple(LoopSegmentType)
        if length <= 0:
            empty = torch.zeros((0,), dtype=torch.int64, device=device)
            return {kind: empty for kind in kinds}
        if not segments:
            zeros = torch.zeros((length,), dtype=torch.int64, device=device)
            return {kind: zeros.clone() for kind in kinds}

        kind_to_index = {kind: idx for idx, kind in enumerate(kinds)}
        kind_ids = torch.tensor([kind_to_index[segment.kind] for segment in segments], dtype=torch.long, device=device)
        starts = torch.tensor([int(segment.start) for segment in segments], dtype=torch.long, device=device)
        stops = torch.tensor([int(segment.stop) for segment in segments], dtype=torch.long, device=device)
        diff = torch.zeros((len(kinds), length + 1), dtype=torch.int32, device=device)
        flat = diff.view(-1)
        stride = length + 1
        ones = torch.ones_like(kind_ids, dtype=flat.dtype)
        flat.scatter_add_(0, kind_ids * stride + starts, ones)
        flat.scatter_add_(0, kind_ids * stride + (stops + 1), -ones)
        masks = (torch.cumsum(diff[:, :-1], dim=1) > 0).to(dtype=torch.int64)
        return {kind: masks[idx] for idx, kind in enumerate(kinds)}

    @staticmethod
    def _segment_nucleotide_mask(segments, segment_type: LoopSegmentType, length: int, device: torch.device) -> Tensor:
        return RnaSecondaryStructureContext._segment_nucleotide_masks(segments, length, device)[segment_type]
