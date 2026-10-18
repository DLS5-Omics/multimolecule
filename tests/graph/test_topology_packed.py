# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import pytest
import torch
from danling import NestedTensor

from multimolecule.graph import DirectedGraph, UndirectedGraph

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize(
    "edges,nodes,expected",
    [
        ([], [], []),
        ([], [-7], [[-7]]),
        ([], [9, -3, 4], [[-3], [4], [9]]),
        ([[4, 1], [7, 9], [9, 10]], [-5], [[-5], [1, 4], [7, 9, 10]]),
        ([[3, 4], [0, 1]], [], [[0, 1], [3, 4]]),
    ],
)
def test_connected_components_keep_packed_values_and_lazy_metadata(device, dtype, graph_type, edges, nodes, expected):
    edge_index = torch.tensor(edges, dtype=dtype, device=device).reshape(-1, 2)
    graph = graph_type(edge_index=edge_index, nodes=edge_index.new_tensor(nodes))

    components = graph.connected_components()

    assert isinstance(components, NestedTensor)
    assert components.concat.dtype == dtype
    assert components.concat.device == edge_index.device
    assert components._offsets.device.type == components._physical_shape.device.type == "cpu"
    if expected:
        assert components._packed_sizes is None
        assert components._element_shapes is None
    assert components._cached_storage is None
    assert len(components) == len(expected)
    assert components.concat.tolist() == [node for component in expected for node in component]
    assert components.tolist() == expected
    if expected:
        assert components[0].tolist() == expected[0]
        assert components[-1].tolist() == expected[-1]
        padded = components.tensor
        for index, component in enumerate(expected):
            assert padded[index, : len(component)].tolist() == component


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_many_components_do_not_split_values_or_create_python_shape_lists(monkeypatch, device, graph_type):
    nodes = torch.arange(4096, dtype=torch.int32, device=device) * 97 - 10**6
    graph = graph_type(edge_index=nodes.new_empty((0, 2)), nodes=nodes)

    def reject_python_components(*args, **kwargs):
        raise AssertionError("Connected components must be constructed directly from packed tensors")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "split", reject_python_components)
        patch.setattr(torch.Tensor, "tolist", reject_python_components)
        patch.setattr(NestedTensor, "__new__", staticmethod(reject_python_components))
        components = graph.connected_components()

    assert len(components) == nodes.numel()
    assert torch.equal(components.concat, nodes)
    assert components._packed_sizes is None
    assert components._element_shapes is None
    assert components._cached_storage is None
    assert components._offsets.shape == (nodes.numel() + 1,)
    assert components._physical_shape.shape == (nodes.numel(), 1)
    assert components[0].tolist() == [int(nodes[0])]
    assert components[-1].tolist() == [int(nodes[-1])]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.int64])
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_dense_node_count_constructor_keeps_isolated_nodes(device, dtype, graph_type):
    edges = torch.tensor([[2, 0], [1, 2]], dtype=dtype, device=device)

    graph = graph_type(edge_index=edges, num_nodes=4)

    assert graph.nodes.tolist() == [0, 1, 2, 3]
    assert graph.nodes.dtype == dtype
    assert graph.nodes.device == edges.device
    assert graph.num_nodes.item() == 4
    assert graph.connected_components().tolist() == [[0, 1, 2], [3]]


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_dense_node_count_constructor_rejects_ambiguous_or_invalid_nodes(graph_type):
    with pytest.raises(ValueError):
        graph_type(nodes=torch.tensor([0, 1]), num_nodes=2)
    with pytest.raises(ValueError):
        graph_type(num_nodes=-1)
    with pytest.raises(TypeError):
        graph_type(num_nodes=True)
    with pytest.raises(ValueError):
        graph_type(edge_index=torch.tensor([[0, 1]], dtype=torch.int8), num_nodes=129)
    with pytest.raises(ValueError):
        graph_type(edge_index=torch.tensor([[-1, 1]]), num_nodes=4)
    with pytest.raises(ValueError):
        graph_type(edge_index=torch.tensor([[0, 4]]), num_nodes=4)


@pytest.mark.parametrize("device", DEVICES)
def test_dense_components_remain_correct_after_nodes_become_sparse(device):
    edges = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    graph = UndirectedGraph(edge_index=edges, num_nodes=3)

    graph.add_edge(2, 99)
    assert graph.connected_components().tolist() == [[0, 1], [2, 99]]
    graph.remove_node(1)
    assert graph.connected_components().tolist() == [[0], [2, 99]]
    graph.reindex_nodes(edges.new_tensor([[99, -7], [2, 10]]))
    assert graph.connected_components().tolist() == [[-7, 10], [0]]
    graph.reindex_nodes(edges.new_tensor([[-7, 0], [0, 1], [10, 2]]))
    assert graph.connected_components().tolist() == [[0, 2], [1]]


@pytest.mark.parametrize("device", DEVICES)
def test_duplicate_explicit_node_ids_do_not_imply_a_dense_range(device):
    graph = UndirectedGraph(
        edge_index=torch.tensor([[0, 3]], device=device), nodes=torch.tensor([0, 2, 2, 3], device=device)
    )

    assert graph.nodes.tolist() == [0, 2, 3]
    assert graph.connected_components().tolist() == [[0, 3], [2]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_nodes", [0, 4])
def test_dense_node_count_without_initial_edges(device, num_nodes):
    graph = UndirectedGraph(num_nodes=num_nodes, device=torch.device(device))

    assert graph.nodes.tolist() == list(range(num_nodes))
    assert graph.nodes.device.type == device
    assert graph.connected_components().tolist() == [[node] for node in range(num_nodes)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize("layout", ["dense", "sparse", "empty"])
def test_connected_components_include_edges_added_through_the_edge_table(device, graph_type, layout):
    if layout == "dense":
        edges = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        graph = graph_type(edge_index=edges, num_nodes=3)
        added = edges.new_tensor([[2, 99]])
        expected = [[0, 1], [2, 99]]
    elif layout == "sparse":
        edges = torch.tensor([[-7, 4]], dtype=torch.int32, device=device)
        graph = graph_type(edge_index=edges, nodes=edges.new_tensor([-100]))
        added = edges.new_tensor([[4, 99]])
        expected = [[-100], [-7, 4, 99]]
    else:
        edges = torch.empty((0, 2), dtype=torch.int32, device=device)
        graph = graph_type(edge_index=edges)
        added = edges.new_tensor([[5, 7]])
        expected = [[5, 7]]

    graph.edges.extend(added)

    assert graph.connected_components().tolist() == expected
