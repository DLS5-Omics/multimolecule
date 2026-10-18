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

import pytest
import torch

from multimolecule.graph import DirectedGraph, EdgeTable, UndirectedGraph


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


def test_mixed_feature_edges_are_rejected():
    g = UndirectedGraph()
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})

    with pytest.raises(ValueError, match="feature must be provided"):
        g.add_edge(1, 2)


def test_edge_attr_pruned_on_remove():
    g = UndirectedGraph()
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.remove_node(0)
    assert g.edge_index.numel() == 0
    assert g.edge_features is None or all(tensor.numel() == 0 for tensor in g.edge_features.values())


def test_edge_table_inplace_prune_is_explicit():
    table = EdgeTable(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        features={"type": torch.tensor([EdgeType.PAIR, EdgeType.BACKBONE])},
    )

    table.prune_(torch.tensor([True, False]))

    assert not hasattr(table, "prune")
    assert table.edge_index.tolist() == [[0, 1]]
    assert table.features is not None
    assert table.features["type"].tolist() == [EdgeType.PAIR]


def test_edge_ids_subgraph_and_coalesce():
    g = UndirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(1, 2, {"type": torch.tensor(EdgeType.BACKBONE)})

    assert g.edge_indices(edge_type=EdgeType.PAIR) == [0, 1]
    sub = g.edge_subgraph(torch.tensor([True, False, True]))
    assert sub.edge_list() == [(0, 1), (1, 2)]

    g.coalesce()
    assert g.edge_list() == [(0, 1), (1, 2)]
    assert g.edge_indices(edge_type=EdgeType.PAIR) == [0]


def test_path_component_methods_are_explicit_only():
    undirected = UndirectedGraph()
    undirected.add_edges([(0, 1), (1, 2)])
    is_path, path = undirected.is_path_component([0, 1, 2])
    assert is_path
    assert path == [0, 1, 2]
    assert not hasattr(undirected, "is_path")

    directed = DirectedGraph()
    directed.add_edges([(0, 1), (1, 2), (2, 0)])
    assert directed.is_path_component([0, 1, 2])
    assert not hasattr(directed, "is_path")
    assert not hasattr(directed, "get_next_pair")
    assert not hasattr(directed, "get_prev_pair")


@pytest.mark.parametrize("column_types", [False, True])
def test_neighbors_are_sorted_unique_with_reversed_edges_and_self_loops(column_types):
    types = torch.tensor([EdgeType.PAIR, EdgeType.BACKBONE, EdgeType.PK, EdgeType.PAIR])
    g = UndirectedGraph(
        edge_index=torch.tensor([[4, -2], [-2, 1], [1, -2], [-2, -2]]),
        edge_features={"type": types[:, None] if column_types else types},
    )

    assert g.neighbors(-2) == [-2, 1, 4]
    assert g.neighbors(-2, edge_type=EdgeType.PAIR) == [-2, 4]
    assert g.neighbors(-2, edge_type=EdgeType.PK) == [1]
    assert g.neighbors(99) == []
    del g.edge_features["type"]
    assert g.neighbors(-2, edge_type=EdgeType.PK) == [-2, 1, 4]


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize(
    "edges,nodes,expected",
    [
        ([], set(), []),
        ([], {-2, 4}, [[-2], [4]]),
        ([[4, 1], [8, 9]], set(), [[1, 4], [8, 9]]),
        ([[4, 1], [1, 4], [1, 1], [8, 9]], {-2, 1, 4, 8, 9}, [[-2], [1, 4], [8, 9]]),
    ],
)
def test_connected_components_preserve_isolated_nodes_and_sorted_order(graph_type, edges, nodes, expected):
    g = graph_type(edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2), nodes=nodes)

    assert g.connected_components() == expected


@pytest.mark.parametrize(
    "edges,component,expected",
    [
        ([], [], (False, [])),
        ([], [7], (False, [])),
        ([[4, 1], [1, -2], [1, 4]], [4, -2, 1], (True, [-2, 1, 4])),
        ([[0, 1], [1, 2], [2, 0]], [0, 1, 2], (False, [])),
        ([[0, 1], [0, 2], [0, 3]], [0, 1, 2, 3], (False, [])),
        ([[0, 1], [2, 3], [3, 4], [4, 2]], [0, 1, 2, 3, 4], (False, [])),
        ([[0, 1], [1, 2], [1, 9]], [0, 1, 2], (False, [])),
    ],
)
def test_undirected_path_component(edges, component, expected):
    g = UndirectedGraph(edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2))

    assert g.is_path_component(component) == expected


@pytest.mark.parametrize(
    "edges,component,path,is_path",
    [
        ([], [0], [0], False),
        ([[0, 1], [1, 2]], [0, 1, 2], [0, 1, 2], False),
        ([[0, 1], [0, 1], [1, 2], [2, 0]], [0, 1, 2], [0, 1, 2], True),
        ([[0, 1], [1, 2], [1, 3]], [0, 1, 2, 3], [0, 1], False),
    ],
)
def test_directed_successor_walk_and_path_component(edges, component, path, is_path):
    g = DirectedGraph(edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2))

    assert g.walk_successor_path(0) == path
    assert g.is_path_component(component) is is_path


@pytest.mark.parametrize("column_types", [False, True])
def test_degree_by_type_counts_unique_neighbors_and_includes_absent_types(column_types):
    types = torch.tensor([EdgeType.PAIR, EdgeType.PAIR, EdgeType.BACKBONE, EdgeType.PK, EdgeType.PK, 3])
    g = UndirectedGraph(
        edge_index=torch.tensor([[0, 1], [1, 0], [0, 2], [0, 2], [0, 0], [8, 9]]),
        edge_features={"type": types[:, None] if column_types else types},
    )

    assert g.degree_by_type(0) == {0: 1, 1: 1, 2: 2, 3: 0}
    assert g.degree_by_type(99) == {0: 0, 1: 0, 2: 0, 3: 0}
    del g.edge_features["type"]
    assert g.degree_by_type(0) == {0: 3}


@pytest.mark.parametrize(
    "graph_type,multi_edges,expected_edges,expected_types",
    [
        (UndirectedGraph, False, [[0, 2], [2, 3]], [10, 24]),
        (UndirectedGraph, True, [[0, 2], [0, 2], [0, 2], [0, 2], [2, 3]], [10, 20, 21, 22, 24]),
        (DirectedGraph, False, [[0, 2], [2, 0], [5, 5], [2, 3]], [10, 20, 23, 24]),
        (DirectedGraph, True, [[0, 2], [2, 0], [0, 2], [2, 0], [5, 5], [2, 3]], [10, 20, 21, 22, 23, 24]),
    ],
)
def test_batch_edges_preserve_order_features_and_node_registration(
    graph_type, multi_edges, expected_edges, expected_types
):
    g = graph_type(
        edge_index=torch.tensor([[0, 2]]),
        nodes={0, 2, 9},
        edge_features={"type": torch.tensor([10]), "vector": torch.tensor([[10, 11]])},
        allow_multi_edges=multi_edges,
    )
    g.add_edges(
        iter([(2, 0), (0, 2), (2, 0), (5, 5), (2, 3)]),
        ({"type": value, "vector": [value, value + 1]} for value in range(20, 25)),
    )

    assert g.edge_index.tolist() == expected_edges
    assert g.edge_features["type"].tolist() == expected_types
    assert g.edge_features["vector"].tolist() == [[value, value + 1] for value in expected_types]
    assert g.nodes == {0, 2, 3, 5, 9}


def test_skipped_batch_edges_do_not_validate_attributes():
    g = UndirectedGraph(edge_index=torch.tensor([[0, 1]]), nodes={0, 1})

    g.add_edges([(1, 0), (9, 9)], [{"ignored": object()}, None])

    assert g.edge_index.tolist() == [[0, 1]]
    assert g.edge_features is None
    assert g.nodes == {0, 1, 9}


@pytest.mark.parametrize("tensor_input", [False, True])
def test_edge_table_extend_aligns_vector_features(tensor_input):
    table = EdgeTable(torch.tensor([[0, 1]]), {"value": torch.tensor([[1, 2]])})
    edges = torch.tensor([[1, 2], [2, 3]])

    table.extend(
        edges if tensor_input else iter(edges),
        iter([{"value": torch.tensor([3, 4])}, {"value": torch.tensor([[5, 6]])}]),
    )

    assert table.edge_index.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert table.features["value"].tolist() == [[1, 2], [3, 4], [5, 6]]


@pytest.mark.parametrize("feature", [None, {"wrong": 1}, {"value": [1, 2, 3]}])
def test_rejected_edge_features_leave_table_unchanged(feature):
    table = EdgeTable(torch.tensor([[0, 1]]), {"value": torch.tensor([[1, 2]])})

    with pytest.raises((ValueError, RuntimeError)):
        table.append(torch.tensor([1, 2]), feature)

    assert table.edge_index.tolist() == [[0, 1]]
    assert table.features["value"].tolist() == [[1, 2]]


def test_new_features_cannot_leave_existing_edges_without_values():
    table = EdgeTable(torch.tensor([[0, 1]]))

    with pytest.raises(ValueError):
        table.append(torch.tensor([1, 2]), {"value": 3})

    assert table.edge_index.tolist() == [[0, 1]]
    assert table.features is None


@pytest.mark.parametrize(
    "mapping,expected_edges,expected_nodes",
    [
        ({}, [[0, 1], [1, 2]], {0, 1, 2, 9}),
        ({0: 1, 1: 0}, [[1, 0], [0, 2]], {0, 1, 2, 9}),
        ({0: 1, 1: 2, 9: -9}, [[1, 2], [2, 2]], {1, 2, -9}),
        ({0: 3, 1: 3, 2: 3, 9: 8}, [[3, 3], [3, 3]], {3, 8}),
    ],
)
def test_reindex_nodes_applies_mapping_to_original_nodes(mapping, expected_edges, expected_nodes):
    g = UndirectedGraph(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
        nodes={0, 1, 2, 9},
        edge_features={"value": torch.tensor([3, 4])},
    )

    g.reindex_nodes(mapping)

    assert g.edge_index.tolist() == expected_edges
    assert g.edge_index.dtype == torch.int32
    assert g.nodes == expected_nodes
    assert g.edge_features["value"].tolist() == [3, 4]


@pytest.mark.parametrize("multi_edges,expected_types", [(False, [10, 40]), (True, [10, 20, 40])])
def test_undirected_conversion_preserves_first_features_and_isolated_nodes(multi_edges, expected_types):
    g = DirectedGraph(
        edge_index=torch.tensor([[2, 0], [0, 2], [1, 1], [2, 3]]),
        nodes={9},
        edge_features={"type": torch.tensor([[10], [20], [30], [40]])},
        allow_multi_edges=multi_edges,
    )

    converted = g.undirected()

    assert converted.edge_index.tolist() == ([[0, 2], [0, 2], [2, 3]] if multi_edges else [[0, 2], [2, 3]])
    assert converted.edge_features["type"].tolist() == expected_types
    assert converted.nodes == {0, 1, 2, 3, 9}
