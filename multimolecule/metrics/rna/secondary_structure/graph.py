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

from torch import Tensor

from .common import approximate_graph_edit_distance, loop_substitution_cost, segment_length_cost
from .context import RnaSecondaryStructureContext


def loop_helix_graph_ged(context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    (
        pred_loops,
        tgt_loops,
        pred_helices,
        tgt_helices,
        pred_edges,
        tgt_edges,
        _loop_map,
        _helix_map,
    ) = context.loop_helix_assignment

    pred_nodes = [("loop", loop) for loop in pred_loops] + [("helix", helix) for helix in pred_helices]
    tgt_nodes = [("loop", loop) for loop in tgt_loops] + [("helix", helix) for helix in tgt_helices]

    pred_edges_labeled: list[tuple[int, int, object]] = []
    tgt_edges_labeled: list[tuple[int, int, object]] = []
    helix_offset_pred = len(pred_loops)
    helix_offset_tgt = len(tgt_loops)
    for loop_idx, helix_idx in pred_edges:
        pred_edges_labeled.append((int(loop_idx), helix_offset_pred + int(helix_idx), ("loop_helix", 0)))
    for loop_idx, helix_idx in tgt_edges:
        tgt_edges_labeled.append((int(loop_idx), helix_offset_tgt + int(helix_idx), ("loop_helix", 0)))

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
        pred_edges_labeled,
        tgt_edges_labeled,
        node_cost_fn=_node_cost,
        device=device,
    )
