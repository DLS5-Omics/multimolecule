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
from random import Random

import pytest
import torch
from danling import NestedTensor

from multimolecule.graph import DirectedGraph, EdgeTable, UndirectedGraph

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


def assert_bool_tensor(value, expected, device):
    assert isinstance(value, torch.Tensor)
    assert value.shape == ()
    assert value.dtype == torch.bool
    assert value.device == device
    assert value.item() is expected


class EdgeType(IntEnum):
    BACKBONE = 0
    PAIR = 1
    PK = 2


def test_edge_attr_kind_filtering_and_subgraph():
    g = UndirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 2, {"type": torch.tensor(EdgeType.BACKBONE)})

    assert g.neighbors(0).tolist() == [1, 2]
    assert g.neighbors(0, edge_type=EdgeType.PAIR).tolist() == [1]
    assert g.edge_list(edge_type=EdgeType.BACKBONE).tolist() == [[0, 2]]

    sub = g.subgraph(torch.tensor([0, 1]))
    assert sub.edge_features is not None
    assert sub.edge_features["type"].numel() == 1
    assert sub.neighbors(0, edge_type=EdgeType.PAIR).tolist() == [1]


def test_directed_kind_filters_and_multi_edges():
    g = DirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.BACKBONE)})
    g.add_edge(0, 2, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 2, {"type": torch.tensor(EdgeType.PK)})

    assert g.successors(0).tolist() == [1, 2]
    assert g.successors(0, edge_type=EdgeType.PAIR).tolist() == [2]
    assert g.successors(0, edge_type=EdgeType.PK).tolist() == [2]


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
    assert table.num_edges.shape == ()
    assert table.num_edges.dtype == torch.long
    assert table.num_edges.device == table.edge_index.device
    assert table.num_edges.item() == 1


def test_edge_ids_subgraph_and_coalesce():
    g = UndirectedGraph(allow_multi_edges=True)
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(0, 1, {"type": torch.tensor(EdgeType.PAIR)})
    g.add_edge(1, 2, {"type": torch.tensor(EdgeType.BACKBONE)})

    assert g.edge_indices(edge_type=EdgeType.PAIR).tolist() == [0, 1]
    sub = g.edge_subgraph(torch.tensor([True, False, True]))
    assert sub.edge_list().tolist() == [[0, 1], [1, 2]]

    g.coalesce()
    assert g.edge_list().tolist() == [[0, 1], [1, 2]]
    assert g.edge_indices(edge_type=EdgeType.PAIR).tolist() == [0]


def test_path_component_methods_are_explicit_only():
    undirected = UndirectedGraph()
    undirected.add_edges(torch.tensor([[0, 1], [1, 2]]))
    is_path, path = undirected.is_path_component([0, 1, 2])
    assert_bool_tensor(is_path, True, undirected.device)
    assert path.tolist() == [0, 1, 2]
    assert not hasattr(undirected, "is_path")

    directed = DirectedGraph()
    directed.add_edges(torch.tensor([[0, 1], [1, 2], [2, 0]]))
    assert_bool_tensor(directed.is_path_component([0, 1, 2]), True, directed.device)
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

    assert g.neighbors(-2).tolist() == [-2, 1, 4]
    assert g.neighbors(-2, edge_type=EdgeType.PAIR).tolist() == [-2, 4]
    assert g.neighbors(-2, edge_type=EdgeType.PK).tolist() == [1]
    assert g.neighbors(99).tolist() == []
    del g.edge_features["type"]
    assert g.neighbors(-2, edge_type=EdgeType.PK).tolist() == [-2, 1, 4]


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize(
    "edges,nodes,expected",
    [
        ([], [], []),
        ([], [-2, 4], [[-2], [4]]),
        ([[4, 1], [8, 9]], [], [[1, 4], [8, 9]]),
        ([[4, 1], [8, 9]], [-2], [[-2], [1, 4], [8, 9]]),
        ([[4, 1], [1, 4], [1, 1], [8, 9]], [-2, 1, 4, 8, 9], [[-2], [1, 4], [8, 9]]),
    ],
)
def test_connected_components_preserve_isolated_nodes_and_sorted_order(graph_type, edges, nodes, expected):
    g = graph_type(
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2), nodes=torch.tensor(nodes, dtype=torch.long)
    )

    components = g.connected_components()

    assert isinstance(components, NestedTensor)
    assert components.tolist() == expected


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

    is_path, path = g.is_path_component(component)

    assert_bool_tensor(is_path, expected[0], g.device)
    assert path.tolist() == expected[1]


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

    assert g.walk_successor_path(0).tolist() == path
    assert_bool_tensor(g.is_path_component(component), is_path, g.device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_connected_components_keep_tensor_dtype_and_device(device, dtype, graph_type):
    edges = torch.tensor([[10**9, -7], [-7, 3], [99, 100]], dtype=dtype, device=device)
    graph = graph_type(edge_index=edges, nodes=edges.new_tensor([-99]))

    components = graph.connected_components()

    assert isinstance(components, NestedTensor)
    assert components.dtype == graph.edge_index.dtype
    assert components.device == edges.device
    assert components.tolist() == [[-99], [-7, 3, 10**9], [99, 100]]
    for component in components:
        assert component.dtype == graph.edge_index.dtype
        assert component.device == edges.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_empty_components_are_nested_tensors(device, graph_type):
    graph = graph_type(edge_index=torch.empty((0, 2), dtype=torch.long, device=device))

    components = graph.connected_components()

    assert isinstance(components, NestedTensor)
    assert len(components) == 0
    assert components.concat.numel() == 0
    assert components.dtype == graph.edge_index.dtype
    assert components.device == graph.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_traversals_do_not_export_edge_table_to_python(monkeypatch, device, dtype):
    edges = torch.tensor([[-7, 3], [3, 99], [99, 10**9]], dtype=dtype, device=device)
    undirected = UndirectedGraph(edge_index=edges, nodes=edges.new_tensor([-99]))
    directed = DirectedGraph(edge_index=edges, nodes=edges.new_tensor([-99]))
    component = edges.new_tensor([-7, 3, 99, 10**9])
    original_tolist = torch.Tensor.tolist

    def reject_edge_export(tensor):
        if tensor.ndim == 2 and tensor.shape[1] == 2:
            raise AssertionError("Traversal must not copy the edge table into a Python list")
        return original_tolist(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", reject_edge_export)
        undirected_components = undirected.connected_components()
        directed_components = directed.connected_components()
        is_path, path = undirected.is_path_component(component)
        walk = directed.walk_successor_path(-7)
        directed_is_path = directed.is_path_component(component)

    assert undirected_components.tolist() == [[-99], [-7, 3, 99, 10**9]]
    assert directed_components.tolist() == [[-99], [-7, 3, 99, 10**9]]
    assert_bool_tensor(is_path, True, edges.device)
    assert_bool_tensor(directed_is_path, False, edges.device)
    for result in (path, walk):
        assert isinstance(result, torch.Tensor)
        assert result.dtype == edges.dtype
        assert result.device == edges.device
        assert torch.equal(result, component)


@pytest.mark.parametrize("device", DEVICES)
def test_long_paths_and_cycles_keep_tensor_outputs(device):
    nodes = torch.arange(2048, dtype=torch.int32, device=device) * 1000 - 10**7
    edges = torch.stack((nodes[:-1], nodes[1:]), dim=1)
    undirected = UndirectedGraph(edge_index=edges)
    directed = DirectedGraph(edge_index=torch.cat((edges, torch.stack((nodes[-1], nodes[0]))[None])))

    is_path, path = undirected.is_path_component(nodes)
    walk = directed.walk_successor_path(nodes[0])

    assert_bool_tensor(is_path, True, nodes.device)
    assert_bool_tensor(directed.is_path_component(nodes), True, nodes.device)
    assert torch.equal(path, nodes)
    assert torch.equal(walk, nodes)
    assert path.dtype == walk.dtype == nodes.dtype
    assert path.device == walk.device == nodes.device


@pytest.mark.parametrize("device", DEVICES)
def test_random_components_and_paths_against_networkx(device):
    nx = pytest.importorskip("networkx")
    random = Random(20261018)
    nodes = [-(10**9), -7, 0, 3, 99, 1000, 10**9]

    for _ in range(12):
        edges = [(u, v) for u in nodes for v in nodes if random.random() < 0.13]
        edges += edges[:3]
        edge_index = torch.tensor(edges, dtype=torch.int32, device=device).reshape(-1, 2)
        reference = nx.Graph()
        reference.add_nodes_from(nodes)
        reference.add_edges_from(edges)
        expected_components = sorted(sorted(component) for component in nx.connected_components(reference))
        undirected = UndirectedGraph(edge_index=edge_index, nodes=edge_index.new_tensor(nodes))
        directed = DirectedGraph(edge_index=edge_index, nodes=edge_index.new_tensor(nodes))

        assert undirected.connected_components().tolist() == expected_components
        assert directed.connected_components().tolist() == expected_components
        for component in expected_components:
            subgraph = reference.subgraph(component)
            expected_is_path = (
                len(component) > 1 and nx.is_tree(subgraph) and max(dict(subgraph.degree()).values()) == 2
            )
            if len(component) == 2:
                expected_is_path = nx.is_tree(subgraph)
            is_path, path = undirected.is_path_component(component)
            assert_bool_tensor(is_path, expected_is_path, edge_index.device)
            if expected_is_path:
                ends = sorted(node for node, degree in subgraph.degree() if degree == 1)
                assert path.tolist() == nx.shortest_path(subgraph, ends[0], ends[1])
            else:
                assert path.numel() == 0

        directed_reference = nx.DiGraph()
        directed_reference.add_nodes_from(nodes)
        directed_reference.add_edges_from(edges)
        start = random.choice(nodes)
        expected_walk = [start]
        for _ in nodes:
            successors = list(directed_reference.successors(expected_walk[-1]))
            if len(successors) != 1 or successors[0] in expected_walk:
                break
            expected_walk.append(successors[0])
        assert directed.walk_successor_path(start).tolist() == expected_walk


@pytest.mark.parametrize("column_types", [False, True])
def test_degree_by_type_counts_unique_neighbors_and_includes_absent_types(column_types):
    types = torch.tensor([EdgeType.PAIR, EdgeType.PAIR, EdgeType.BACKBONE, EdgeType.PK, EdgeType.PK, 3])
    g = UndirectedGraph(
        edge_index=torch.tensor([[0, 1], [1, 0], [0, 2], [0, 2], [0, 0], [8, 9]]),
        edge_features={"type": types[:, None] if column_types else types},
    )

    assert g.degree_by_type(0).tolist() == [[0, 1], [1, 1], [2, 2], [3, 0]]
    assert g.degree_by_type(99).tolist() == [[0, 0], [1, 0], [2, 0], [3, 0]]
    del g.edge_features["type"]
    assert g.degree_by_type(0).tolist() == [[0, 3]]


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
        nodes=torch.tensor([0, 2, 9]),
        edge_features={"type": torch.tensor([10]), "vector": torch.tensor([[10, 11]])},
        allow_multi_edges=multi_edges,
    )
    g.add_edges(
        torch.tensor([[2, 0], [0, 2], [2, 0], [5, 5], [2, 3]]),
        {"type": torch.arange(20, 25), "vector": torch.stack((torch.arange(20, 25), torch.arange(21, 26)), dim=1)},
    )

    assert g.edge_index.tolist() == expected_edges
    assert g.edge_features["type"].tolist() == expected_types
    assert g.edge_features["vector"].tolist() == [[value, value + 1] for value in expected_types]
    assert g.nodes.tolist() == [0, 2, 3, 5, 9]


def test_skipped_batch_edges_do_not_validate_attributes():
    g = UndirectedGraph(edge_index=torch.tensor([[0, 1]]), nodes=torch.tensor([0, 1]))

    g.add_edges(torch.tensor([[1, 0], [9, 9]]), {"ignored": torch.tensor([1, 2])})

    assert g.edge_index.tolist() == [[0, 1]]
    assert g.edge_features is None
    assert g.nodes.tolist() == [0, 1, 9]


def test_edge_table_extend_aligns_vector_features():
    table = EdgeTable(torch.tensor([[0, 1]]), {"value": torch.tensor([[1, 2]])})
    edges = torch.tensor([[1, 2], [2, 3]])

    table.extend(edges, {"value": torch.tensor([[3, 4], [5, 6]])})

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
        ([], [[0, 1], [1, 2]], [0, 1, 2, 9]),
        ([[0, 1], [1, 0]], [[0, 1], [0, 2]], [0, 1, 2, 9]),
        ([[0, 1], [1, 2], [9, -9]], [[1, 2], [2, 2]], [-9, 1, 2]),
        ([[0, 3], [1, 3], [2, 3], [9, 8]], [[3, 3], [3, 3]], [3, 8]),
    ],
)
def test_reindex_nodes_applies_mapping_to_original_nodes(mapping, expected_edges, expected_nodes):
    g = UndirectedGraph(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
        nodes=torch.tensor([0, 1, 2, 9], dtype=torch.int32),
        edge_features={"value": torch.tensor([3, 4])},
    )

    g.reindex_nodes(torch.tensor(mapping, dtype=torch.int32).reshape(-1, 2))

    assert g.edge_index.tolist() == expected_edges
    assert g.edge_index.dtype == torch.int32
    assert g.nodes.tolist() == expected_nodes
    assert g.edge_features["value"].tolist() == [3, 4]


@pytest.mark.parametrize("multi_edges,expected_types", [(False, [10, 40]), (True, [10, 20, 40])])
def test_undirected_conversion_preserves_first_features_and_isolated_nodes(multi_edges, expected_types):
    g = DirectedGraph(
        edge_index=torch.tensor([[2, 0], [0, 2], [1, 1], [2, 3]]),
        nodes=torch.tensor([9]),
        edge_features={"type": torch.tensor([[10], [20], [30], [40]])},
        allow_multi_edges=multi_edges,
    )

    converted = g.undirected()

    assert converted.edge_index.tolist() == ([[0, 2], [0, 2], [2, 3]] if multi_edges else [[0, 2], [2, 3]])
    assert converted.edge_features["type"].tolist() == [[value] for value in expected_types]
    assert converted.nodes.tolist() == [0, 1, 2, 3, 9]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_graph_queries_and_mutations_keep_tensors_on_device(monkeypatch, device, dtype, graph_type):
    edges = torch.tensor([[4, -2], [-2, 1], [1, -2], [-2, -2], [99, 100]], dtype=dtype, device=device)
    features = {"type": edges.new_tensor([[1], [0], [2], [1], [3]])}
    graph = graph_type(edge_index=edges, nodes=edges.new_tensor([-100]), edge_features=features)

    def reject_export(*args, **kwargs):
        raise AssertionError("Graph operations must not export tensor data to Python or CPU")

    with monkeypatch.context() as patch:
        for method in ("tolist", "item", "cpu", "numpy"):
            patch.setattr(torch.Tensor, method, reject_export)
        node_count = graph.num_nodes
        found = graph.has_edge(edges.new_tensor(4), edges.new_tensor(-2))
        missing = graph.has_edge(edges.new_tensor(-100), edges.new_tensor(100))
        edge_ids = graph.edge_indices(edge_type=1)
        filtered = graph.edge_list(edge_type=1)
        subgraph = graph.edge_subgraph(edges.new_tensor([True, False, False, True, False], dtype=torch.bool))
        if graph_type is UndirectedGraph:
            neighbors = graph.neighbors(edges.new_tensor(-2))
            degree = graph.degree(edges.new_tensor(-2))
            by_type = graph.degree_by_type(edges.new_tensor(-2))
            induced = graph.subgraph(edges.new_tensor([-100, -2, 4]))
        else:
            successors = graph.successors(edges.new_tensor(-2))
            predecessors = graph.predecessors(edges.new_tensor(-2))
            converted = graph.undirected()
        graph.coalesce()
        graph.add_nodes(edges.new_tensor([700, 700]))
        graph.add_edges(edges.new_tensor([[100, 700]]), attrs={"type": edges.new_tensor([[4]])})
        graph.reindex_nodes(edges.new_tensor([[99, 1000], [100, 99]]))

    assert node_count.shape == ()
    assert node_count.dtype == torch.long
    assert node_count.device == edges.device
    assert node_count.item() == 6
    assert_bool_tensor(found, True, edges.device)
    assert_bool_tensor(missing, False, edges.device)
    assert edge_ids.tolist() == [0, 3]
    assert edge_ids.dtype == torch.long
    assert filtered.tolist() == ([[-2, 4], [-2, -2]] if graph_type is UndirectedGraph else [[4, -2], [-2, -2]])
    assert subgraph.edge_features["type"].tolist() == [[1], [1]]
    assert graph.nodes.tolist() == [-100, -2, 1, 4, 99, 700, 1000]
    assert graph.edge_index[-1].tolist() == [99, 700]
    assert graph.edge_features["type"][-1].tolist() == [4]
    if graph_type is UndirectedGraph:
        assert neighbors.tolist() == [-2, 1, 4]
        assert degree.shape == ()
        assert degree.dtype == torch.long
        assert degree.device == edges.device
        assert degree.item() == 3
        assert by_type.tolist() == [[0, 1], [1, 2], [2, 1], [3, 0]]
        assert induced.nodes.tolist() == [-100, -2, 4]
        assert induced.edge_index.tolist() == [[-2, 4], [-2, -2]]
    else:
        assert successors.tolist() == [-2, 1]
        assert predecessors.tolist() == [-2, 1, 4]
        assert converted.edge_index.tolist() == [[-2, 4], [-2, 1], [99, 100]]
        assert converted.edge_features["type"].tolist() == [[1], [0], [3]]
    for result in (graph.nodes, graph.edge_index, filtered, subgraph.nodes, subgraph.edge_index):
        assert isinstance(result, torch.Tensor)
        assert result.dtype == dtype
        assert result.device == edges.device
    assert edge_ids.device == edges.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "features",
    [None, {"wrong": torch.ones(2)}, {"value": torch.ones(2, 3)}, {"value": torch.ones(1, 2)}],
)
def test_rejected_batch_features_leave_nodes_edges_and_features_unchanged(device, features):
    edges = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    graph = UndirectedGraph(
        edge_index=edges,
        nodes=edges.new_tensor([9]),
        edge_features={"value": torch.tensor([[1, 2]], device=device)},
    )
    nodes_before = graph.nodes.clone()
    edges_before = graph.edge_index.clone()
    features_before = graph.edge_features["value"].clone()
    batch_features = None if features is None else {key: value.to(device) for key, value in features.items()}

    with pytest.raises((ValueError, RuntimeError)):
        graph.add_edges(edges.new_tensor([[1, 2], [2, 3]]), attrs=batch_features)

    assert torch.equal(graph.nodes, nodes_before)
    assert torch.equal(graph.edge_index, edges_before)
    assert torch.equal(graph.edge_features["value"], features_before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_empty_graph_queries_keep_tensor_shapes_dtype_and_device(device, graph_type):
    edges = torch.empty((0, 2), dtype=torch.int32, device=device)
    graph = graph_type(edge_index=edges, nodes=edges.new_tensor([-7, 99]))

    assert graph.nodes.tolist() == [-7, 99]
    assert graph.nodes.dtype == edges.dtype
    assert graph.num_nodes.item() == 2
    assert graph.edge_list().shape == (0, 2)
    assert graph.edge_list().dtype == edges.dtype
    assert graph.edge_indices().shape == (0,)
    assert graph.edge_indices().dtype == torch.long
    assert_bool_tensor(graph.has_edge(-7, 99), False, edges.device)
    if graph_type is UndirectedGraph:
        assert graph.neighbors(-7).shape == (0,)
        assert graph.neighbors(-7).dtype == edges.dtype
        assert graph.degree(-7).item() == 0
        assert graph.degree_by_type(-7).tolist() == [[0, 0]]
        induced = graph.subgraph(edges.new_tensor([-7]))
        assert induced.nodes.tolist() == [-7]
        graph.remove_node(-7)
        assert graph.nodes.tolist() == [99]
    else:
        assert graph.successors(-7).shape == graph.predecessors(-7).shape == (0,)
        assert graph.successors(-7).dtype == graph.predecessors(-7).dtype == edges.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_graph_to_moves_isolated_nodes_and_aligned_features(device):
    graph = UndirectedGraph(
        edge_index=torch.tensor([[0, 1]], dtype=torch.int32),
        nodes=torch.tensor([-7], dtype=torch.int32),
        edge_features={"value": torch.tensor([[3, 4]])},
    )

    graph.to(torch.device(device))

    assert graph.edge_index.device == graph.nodes.device == graph.edge_features["value"].device
    assert graph.nodes.device.type == device
    assert graph.nodes.tolist() == [-7, 0, 1]
    assert graph.nodes.dtype == torch.int32


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize("operation", ["add_edges", "reindex_nodes"])
def test_out_of_range_int64_ids_are_rejected_without_mutating_int32_graph(device, graph_type, operation):
    edges = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    graph = graph_type(edge_index=edges, nodes=edges.new_tensor([9]), edge_features={"value": edges.new_tensor([3])})
    nodes_before = graph.nodes.clone()
    edges_before = graph.edge_index.clone()
    features_before = graph.edge_features["value"].clone()
    invalid_ids = torch.tensor([[2**32, 2]], dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match="out of range"):
        if operation == "add_edges":
            graph.add_edges(invalid_ids, attrs={"value": edges.new_tensor([4])})
        else:
            graph.reindex_nodes(invalid_ids)

    assert torch.equal(graph.nodes, nodes_before)
    assert torch.equal(graph.edge_index, edges_before)
    assert torch.equal(graph.edge_features["value"], features_before)


@pytest.mark.parametrize("device", DEVICES)
def test_undirected_path_must_contain_the_requested_component(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [10, 11], [11, 12]], device=device)
    graph = UndirectedGraph(edge_index=edges)

    is_path, path = graph.is_path_component(edges.new_tensor([0, 2, 3, 11]))

    assert_bool_tensor(is_path, False, edges.device)
    assert path.shape == (0,)
    assert path.dtype == edges.dtype
    assert path.device == edges.device


def test_empty_feature_dictionary_is_treated_as_no_features():
    table = EdgeTable(torch.tensor([[0, 1]]), features={})

    table.extend(torch.tensor([[1, 2]]), features={})

    assert table.features is None
    assert table.edge_index.tolist() == [[0, 1], [1, 2]]


def test_feature_schema_is_retained_without_initial_edges():
    graph = UndirectedGraph(edge_features={"value": torch.empty((0, 2), dtype=torch.long)})

    graph.add_edge(0, 1, attr={"value": torch.tensor([3, 4])})

    assert graph.edge_index.tolist() == [[0, 1]]
    assert graph.edge_features["value"].tolist() == [[3, 4]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_signed_and_unsigned_id_casts_validate_only_representable_bounds(device, dtype):
    graph = UndirectedGraph(edge_index=torch.tensor([[0, 1]], dtype=dtype, device=device))
    graph.add_nodes(torch.tensor([2], dtype=torch.int64, device=device))
    graph.reindex_nodes(torch.tensor([[0, 3]], dtype=torch.int64, device=device))

    assert graph.nodes.tolist() == [1, 2, 3]
    assert graph.edge_index.tolist() == [[1, 3]]
    assert_bool_tensor(graph.has_edge(1, 3), True, graph.device)
    if dtype == torch.uint8:
        with pytest.raises(ValueError, match="out of range"):
            graph.has_edge(-1, 3)
