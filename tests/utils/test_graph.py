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


from enum import IntEnum

import torch

from multimolecule.utils.graph import DirectedGraph, UndirectedGraph


class EdgeType(IntEnum):
    BACKBONE = 0
    PAIR = 1
    PK = 2


def test_edge_attr_kind_filtering_and_subgraph():
    g = UndirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 2, {"type": torch.tensor(EdgeType.BACKBONE)})

    assert set(g.neighbors(0)) == {1, 2}
    assert set(g.neighbors(0, edge_type=EdgeType.PAIR)) == {1}
    assert g.edge_list(edge_type=EdgeType.BACKBONE) == [(0, 2)]

    sub = g.subgraph([0, 1])
    assert sub.edge_features is not None
    assert sub.edge_features["type"].numel() == 1
    assert sub.neighbors(0, edge_type=EdgeType.PAIR) == [1]


def test_directed_kind_filters_and_multi_edges():
    g = DirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.BACKBONE)})
    g.add_edge(0, 2, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 2, {"type": torch.tensor(EdgeType.PK)})

    assert g.successors(0) == [1, 2]
    assert g.successors(0, edge_type=EdgeType.PAIR) == [2]
    assert g.successors(0, edge_type=EdgeType.PK) == [2]


def test_edge_attr_pruned_on_remove():
    g = UndirectedGraph()
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.remove_node(0)
    assert g.edge_index.numel() == 0
    assert g.edge_features is None or all(tensor.numel() == 0 for tensor in g.edge_features.values())


def test_edge_ids_subgraph_and_coalesce():
    g = UndirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(1, 2, {"type": torch.tensor(EdgeType.BACKBONE)})

    assert g.edge_ids(edge_type=EdgeType.PAIR) == [0, 1]
    sub = g.subgraph_by_edge_mask(torch.tensor([True, False, True]))
    assert sub.edge_list() == [(0, 1), (1, 2)]

    g.coalesce()
    assert g.edge_list() == [(0, 1), (1, 2)]
    assert g.edge_ids(edge_type=EdgeType.PAIR) == [0]
