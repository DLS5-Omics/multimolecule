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

import copy
import pickle

import pytest
import torch
from danling import NestedTensor

from multimolecule.graph import DirectedGraph, EdgeTable, UndirectedGraph


@pytest.fixture(params=[UndirectedGraph, DirectedGraph], ids=["undirected", "directed"])
def graph_type(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["flat-types", "column-types"])
def typed_graph(request, device):
    edges = torch.tensor([[0, 1], [1, 0], [0, 2], [0, 2], [0, 0], [8, 9]], dtype=torch.int32, device=device)
    types = edges.new_tensor([1, 1, 0, 2, 2, 3])
    return UndirectedGraph(
        edges, nodes=edges.new_tensor([-7]), edge_features={"type": types[:, None] if request.param else types}
    )


class TestEdgeTable:
    def test_mutations_and_selection_align_features(self, device):
        edges = torch.tensor([[1, 0], [9, 9]], device=device)
        table = EdgeTable(edges, {"value": edges.new_tensor([[1, 2], [3, 4]])})
        table.append(edges.new_tensor([1, 0]), {"value": [5, 6]})
        table.extend(edges.new_tensor([[2, 3]]), {"value": edges.new_tensor([[7, 8]])})
        mask = edges.new_tensor([True, False, True, False], dtype=torch.bool)
        selected = table.masked_select(mask)
        table.prune_(mask)

        assert table.edge_index.tolist() == selected.edge_index.tolist() == [[1, 0], [1, 0]]
        assert table.features["value"].tolist() == selected.features["value"].tolist() == [[1, 2], [5, 6]]
        assert table.num_edges.item() == len(table) == 2

    def test_incompatible_features_leave_data_unchanged(self):
        table = EdgeTable(torch.tensor([[0, 1]]), {"value": torch.tensor([[3, 4]])})
        with pytest.raises(ValueError, match="feature"):
            table.append(torch.tensor([1, 2]), None)
        assert table.edge_index.tolist() == [[0, 1]]
        assert table.features["value"].tolist() == [[3, 4]]


class TestGraph:
    @pytest.mark.parametrize(
        "edges,nodes,expected",
        [
            ([], [], []),
            ([], [-7, 9], [[-7], [9]]),
            ([[4, 1], [7, 9], [9, 10], [9, 7]], [-5], [[-5], [1, 4], [7, 9, 10]]),
            ([[2, 3], [0, 3]], [1], [[0, 2, 3], [1]]),
        ],
    )
    def test_components_include_isolates_and_ignore_edge_direction(self, graph_type, device, edges, nodes, expected):
        edge_index = torch.tensor(edges, dtype=torch.int32, device=device).reshape(-1, 2)
        graph = graph_type(edge_index, nodes=edge_index.new_tensor(nodes))
        components = graph.connected_components()

        assert isinstance(components, NestedTensor)
        assert components.tolist() == expected
        assert components.dtype == edge_index.dtype
        assert components.device == edge_index.device

    def test_dense_nodes_include_isolates(self, graph_type, device):
        edges = torch.tensor([[2, 0], [1, 2]], device=device)
        graph = graph_type(edges, num_nodes=4)
        assert graph.nodes.tolist() == [0, 1, 2, 3]
        assert graph.num_nodes.item() == 4
        assert graph.connected_components().tolist() == [[0, 1, 2], [3]]

    @pytest.mark.parametrize("multi_edges", [False, True])
    def test_insertion_preserves_edge_order_and_aligned_features(self, graph_type, device, multi_edges):
        edges = torch.tensor([[0, 2]], device=device)
        graph = graph_type(
            edges,
            nodes=edges.new_tensor([9]),
            edge_features={"value": edges.new_tensor([10])},
            allow_multi_edges=multi_edges,
        )
        graph.add_edges(
            edges.new_tensor([[2, 0], [0, 2], [5, 5], [2, 3]]), {"value": edges.new_tensor([20, 21, 22, 23])}
        )
        if graph_type is UndirectedGraph:
            expected_edges = [[0, 2], [0, 2], [0, 2], [2, 3]] if multi_edges else [[0, 2], [2, 3]]
            expected_values = [10, 20, 21, 23] if multi_edges else [10, 23]
        else:
            expected_edges = (
                [[0, 2], [2, 0], [0, 2], [5, 5], [2, 3]] if multi_edges else [[0, 2], [2, 0], [5, 5], [2, 3]]
            )
            expected_values = [10, 20, 21, 22, 23] if multi_edges else [10, 20, 22, 23]
        assert graph.edge_index.tolist() == expected_edges
        assert graph.edge_features["value"].tolist() == expected_values
        assert graph.nodes.tolist() == [0, 2, 3, 5, 9]

    def test_edge_table_mutations_are_visible_and_keep_registered_nodes(self, graph_type, device):
        edges = torch.tensor([[0, 1]], device=device)
        graph = graph_type(edges, nodes=edges.new_tensor([9]))
        assert graph.has_edge(0, 1).item() is True
        table = graph.edges
        table.append(edges.new_tensor([1, 2]), None)
        table.prune_(edges.new_tensor([True, False], dtype=torch.bool))
        assert graph.has_edge(1, 2).item() is False
        assert graph.connected_components().tolist() == [[0, 1], [2], [9]]

        graph.edges.edge_index = edges.new_tensor([[30, 20]])
        assert graph.has_edge(30, 20).item() is True
        assert graph.nodes.tolist() == [0, 1, 2, 9, 20, 30]
        graph.reindex_nodes(edges.new_tensor([[20, -2], [30, -3]]))
        assert graph.has_edge(-3, -2).item() is True
        assert graph.nodes.tolist() == [-3, -2, 0, 1, 2, 9]

    def test_reindexing_is_simultaneous_and_retains_features(self, graph_type, device):
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device=device)
        graph = graph_type(edges, nodes=edges.new_tensor([9]), edge_features={"value": edges.new_tensor([3, 4])})
        graph.reindex_nodes(edges.new_tensor([[0, 1], [1, 0], [9, -9]]))
        assert graph.edge_index.tolist() == ([[0, 1], [0, 2]] if graph_type is UndirectedGraph else [[1, 0], [0, 2]])
        assert graph.nodes.tolist() == [-9, 0, 1, 2]
        assert graph.edge_features["value"].tolist() == [3, 4]
        assert graph.edge_index.dtype == torch.int32

    def test_move_preserves_isolates_and_features(self, graph_type, device):
        graph = graph_type(
            torch.tensor([[0, 1]]), nodes=torch.tensor([-7]), edge_features={"value": torch.tensor([[3, 4]])}
        )
        graph.to(device)
        assert graph.edge_index.device == graph.nodes.device == graph.edge_features["value"].device == device
        assert graph.nodes.tolist() == [-7, 0, 1]
        assert graph.edge_features["value"].tolist() == [[3, 4]]

    @pytest.mark.parametrize("duplicate", [copy.copy, copy.deepcopy, pickle.loads], ids=["copy", "deepcopy", "pickle"])
    def test_copy_and_pickle_preserve_queries_and_future_mutations(self, graph_type, duplicate):
        graph = graph_type(torch.tensor([[0, 1]]), nodes=torch.tensor([9]))
        graph.has_edge(0, 1)
        restored = duplicate(pickle.dumps(graph)) if duplicate is pickle.loads else duplicate(graph)
        restored.edges.extend(torch.tensor([[1, 2]]))
        assert restored.nodes.tolist() == [0, 1, 2, 9]
        assert restored.has_edge(1, 2).item() is True
        assert graph.has_edge(1, 2).item() is (duplicate is copy.copy)


class TestUndirectedGraph:
    def test_typed_queries_count_sorted_unique_neighbors(self, typed_graph):
        graph = typed_graph
        assert graph.neighbors(0).tolist() == [0, 1, 2]
        assert graph.neighbors(0, edge_type=1).tolist() == [1]
        assert graph.neighbors(0, edge_type=2).tolist() == [0, 2]
        assert graph.neighbors(-7).tolist() == []
        assert graph.has_edge(1, 0).item() is True
        assert graph.has_edge(1, 2).item() is False
        assert graph.degree(0).item() == 3
        assert graph.degree_by_type(0).tolist() == [[0, 1], [1, 1], [2, 2], [3, 0]]

        degrees = graph.degrees([[0, 8, 0], [99, 1, 2]])
        assert degrees.tolist() == [[3, 1, 3], [0, 1, 1]]
        assert degrees.dtype == torch.long
        assert degrees.device == graph.device
        assert graph.degrees().tolist() == [0, 3, 1, 1, 1, 1]
        assert graph.degrees([0, 1, 2], edge_type=2).tolist() == [2, 0, 1]

    def test_type_queries_use_updated_scalar_tensor_value(self):
        graph = UndirectedGraph(torch.tensor([[0, 1], [0, 2]]), edge_features={"type": torch.tensor([1, 2])})
        edge_type = torch.tensor(1)
        assert graph.neighbors(0, edge_type).tolist() == [1]
        assert graph.degrees([0, 1], edge_type).tolist() == [1, 1]
        edge_type.fill_(2)
        assert graph.neighbors(0, edge_type).tolist() == [2]
        assert graph.degrees([0, 1], edge_type).tolist() == [1, 0]

    def test_subgraphs_and_coalesce_preserve_features(self, typed_graph):
        graph = typed_graph
        assert graph.edge_indices(edge_type=1).tolist() == [0, 1]
        assert graph.edge_list(edge_type=1).tolist() == [[0, 1], [0, 1]]
        selected = graph.edge_subgraph(graph.edge_mask(edge_type=1))
        assert selected.nodes.tolist() == graph.nodes.tolist()
        assert selected.edge_features["type"].reshape(-1).tolist() == [1, 1]
        induced = graph.subgraph(graph.nodes.new_tensor([-7, 0, 1]))
        assert induced.nodes.tolist() == [-7, 0, 1]
        assert induced.edge_index.tolist() == [[0, 1], [0, 1], [0, 0]]
        graph.coalesce()
        assert graph.edge_index.tolist() == [[0, 1], [0, 2], [0, 0], [8, 9]]
        assert graph.edge_features["type"].reshape(-1).tolist() == [1, 0, 2, 3]

    def test_remove_node_prunes_incident_features(self, typed_graph):
        graph = typed_graph
        graph.remove_node(0)
        assert graph.nodes.tolist() == [-7, 1, 2, 8, 9]
        assert graph.edge_index.tolist() == [[8, 9]]
        assert graph.edge_features["type"].reshape(-1).tolist() == [3]
        assert graph.neighbors(1).tolist() == []

    def test_synchronize_makes_direct_edits_visible_to_shared_graphs(self, device):
        edges = torch.tensor([[0, 1]], device=device)
        graph = UndirectedGraph(edges, edge_features={"type": edges.new_tensor([1])})
        other = copy.copy(graph)
        assert other.degrees([0, 1], 1).tolist() == [1, 1]
        assert other.degree_by_type(0).tolist() == [[1, 1]]
        graph.edge_index[0] = edges.new_tensor([3, 2])
        graph.edge_features["type"][0] = 2
        graph.synchronize_()
        assert graph.edge_index.tolist() == [[2, 3]]
        assert graph.nodes.tolist() == [0, 1, 2, 3]
        assert other.neighbors(0).tolist() == []
        assert other.degrees([0, 1, 2, 3], 1).tolist() == [0, 0, 0, 0]
        assert other.degrees([0, 1, 2, 3], 2).tolist() == [0, 0, 1, 1]
        assert other.degree_by_type(2).tolist() == [[2, 1]]

    def test_add_nodes_and_single_edge_preserve_feature_schema(self, device):
        graph = UndirectedGraph(num_nodes=2, device=device, edge_features={"value": torch.empty((0, 2), device=device)})
        graph.add_nodes(torch.tensor([1, -7, -7], device=device))
        graph.add_node(9)
        graph.add_edge(-7, 1, {"value": [3.0, 4.0]})
        assert graph.nodes.tolist() == [-7, 0, 1, 9]
        assert graph.edge_features["value"].tolist() == [[3.0, 4.0]]
        assert graph.degree_by_type(-7).tolist() == [[0, 1]]

    @pytest.mark.parametrize(
        "edges,component,expected",
        [
            ([], [], []),
            ([[4, 1], [1, -2], [1, 4]], [4, -2, 1], [-2, 1, 4]),
            ([[0, 1], [1, 2], [2, 0]], [0, 1, 2], []),
            ([[0, 1], [0, 2], [0, 3]], [0, 1, 2, 3], []),
            ([[0, 1], [1, 2], [1, 9]], [0, 1, 2], []),
            ([[0, 1], [1, 2], [10, 11]], [0, 1, 11], []),
        ],
    )
    def test_path_components(self, device, edges, component, expected):
        edge_index = torch.tensor(edges, dtype=torch.int32, device=device).reshape(-1, 2)
        valid, path = UndirectedGraph(edge_index).is_path_component(component)
        assert valid.item() is bool(expected)
        assert valid.dtype == torch.bool and valid.device == edge_index.device
        assert path.tolist() == expected
        assert path.dtype == edge_index.dtype
        assert path.device == edge_index.device


class TestDirectedGraph:
    def test_typed_successors_and_predecessors(self, device):
        edges = torch.tensor([[0, 1], [0, 2], [2, 0], [2, 2], [0, 2]], device=device)
        graph = DirectedGraph(edges, edge_features={"type": edges.new_tensor([0, 1, 1, 2, 2])})
        assert graph.successors(0).tolist() == [1, 2]
        assert graph.successors(0, 1).tolist() == [2]
        assert graph.predecessors(2).tolist() == [0, 2]
        assert graph.predecessors(2, 1).tolist() == [0]
        assert graph.successors(99).tolist() == graph.predecessors(99).tolist() == []
        assert graph.has_edge(0, 1).item() is True
        assert graph.has_edge(1, 0).item() is False

    @pytest.mark.parametrize("multi_edges", [False, True])
    def test_undirected_conversion_preserves_first_features_and_isolates(self, device, multi_edges):
        edges = torch.tensor([[2, 0], [0, 2], [1, 1], [2, 3]], device=device)
        graph = DirectedGraph(
            edges,
            nodes=edges.new_tensor([9]),
            allow_multi_edges=multi_edges,
            edge_features={"value": edges.new_tensor([10, 20, 30, 40])},
        )
        converted = graph.undirected()
        assert converted.edge_index.tolist() == ([[0, 2], [0, 2], [2, 3]] if multi_edges else [[0, 2], [2, 3]])
        assert converted.edge_features["value"].tolist() == ([10, 20, 40] if multi_edges else [10, 40])
        assert converted.nodes.tolist() == [0, 1, 2, 3, 9]

    def test_walk_stops_at_cycle_branch_or_dead_end_and_observes_mutation(self, device):
        edges = torch.tensor([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5]], dtype=torch.int32, device=device)
        graph = DirectedGraph(edges)
        cycle = graph.walk_successor_path(0)
        assert cycle.tolist() == [0, 1, 2]
        assert cycle.dtype == edges.dtype and cycle.device == edges.device
        assert graph.walk_successor_path(1).tolist() == [1, 2, 0]
        assert graph.walk_successor_path(3).tolist() == [3, 4, 5]
        assert graph.walk_successor_path(99).tolist() == [99]
        graph.add_edge(1, 99)
        assert graph.walk_successor_path(0).tolist() == [0, 1]

    @pytest.mark.parametrize(
        "extra,expected", [([], False), ([[2, 0]], True), ([[2, 99]], False), ([[2, 99], [2, 100]], True)]
    )
    def test_path_component_requires_closed_walk_or_terminal_branch(self, device, extra, expected):
        edges = torch.tensor([[0, 1], [1, 2], *extra], device=device)
        assert DirectedGraph(edges).is_path_component([0, 1, 2]).item() is expected
