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

from typing import TYPE_CHECKING, Tuple

from torch import Tensor

from .common import bipartite_match, f1_from_confusion, precision_from_confusion, recall_from_confusion
from .stems import stem_edge_adjacency_by_type, stem_pair_data_from_segments, stem_score_matrix

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def _stem_assignment_and_edges(context: RnaSecondaryStructureContext) -> Tuple[int, int, int, int, int, int]:
    base = context.length + 1
    device = context.device

    pred_segments, pred_edges = context.pred_topology.stem_graph_components()
    tgt_segments, tgt_edges = context.target_topology.stem_graph_components()

    pred_nodes_total = len(pred_segments)
    tgt_nodes_total = len(tgt_segments)
    pred_edge_set = set(pred_edges)
    tgt_edge_set = set(tgt_edges)
    pred_edges_total = len(pred_edge_set)
    tgt_edges_total = len(tgt_edge_set)
    matched_nodes = 0
    matched_edges = 0

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
        matched_pred, matched_tgt = bipartite_match(score_mat)
        if matched_pred.numel() != 0:
            pred_map = {
                int(pred_idx): int(tgt_idx) for pred_idx, tgt_idx in zip(matched_pred.tolist(), matched_tgt.tolist())
            }
            matched_nodes = len(pred_map)
            for src, dst, edge_type in pred_edge_set:
                tgt_src = pred_map.get(src)
                tgt_dst = pred_map.get(dst)
                if tgt_src is None or tgt_dst is None:
                    continue
                if (tgt_src, tgt_dst, edge_type) in tgt_edge_set:
                    matched_edges += 1

    return (
        matched_nodes,
        pred_nodes_total,
        tgt_nodes_total,
        matched_edges,
        pred_edges_total,
        tgt_edges_total,
    )


def topology_f1(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.topology_confusion
    return f1_from_confusion(cm, context.device)


def topology_precision(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.topology_confusion
    return precision_from_confusion(cm, context.device)


def topology_recall(context: RnaSecondaryStructureContext) -> Tensor:
    cm = context.topology_confusion
    return recall_from_confusion(cm, context.device)


def topology_ged(context: RnaSecondaryStructureContext) -> Tensor:
    return context.topology_ged
