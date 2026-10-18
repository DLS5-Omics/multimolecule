# MultiMolecule
# Copyright (C) 2024-Present MultiMolecule

from __future__ import annotations

import pytest
import torch

from multimolecule.graph import DirectedGraph, UndirectedGraph
from multimolecule.graph.topology import _component_labels_cpu


def _labels_from_components(components, num_nodes):
    values = components.concat.to(torch.long)
    offsets = components._offsets
    lengths = offsets[1:] - offsets[:-1]
    roots = values.index_select(0, offsets[:-1])
    labels = torch.empty(num_nodes, dtype=torch.long)
    labels.index_copy_(0, values, torch.repeat_interleave(roots, lengths))
    return labels


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
def test_cpu_contraction_does_not_stop_at_a_temporary_idempotent_forest(graph_type):
    edges = torch.tensor([[2, 3], [0, 3]])

    # The first hook creates [0,1,2,0], which is idempotent but not finished.
    assert torch.equal(_component_labels_cpu(edges, 4), torch.tensor([0, 1, 0, 0]))
    graph = graph_type(edge_index=edges, num_nodes=4)
    assert graph.connected_components().tolist() == [[0, 2, 3], [1]]


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shuffled", [False, True])
def test_cpu_contraction_compresses_long_paths_and_preserves_isolates(graph_type, dtype, shuffled):
    size = 4096
    nodes = torch.arange(size, dtype=torch.long)
    if shuffled:
        nodes = nodes[torch.randperm(size, generator=torch.Generator().manual_seed(17))]
    edges = torch.stack((nodes[:-1], nodes[1:]), dim=1).to(dtype)
    graph = graph_type(edge_index=edges, num_nodes=size + 2)

    components = graph.connected_components()

    assert torch.equal(components.concat, torch.arange(size + 2, dtype=dtype))
    assert torch.equal(components._offsets, torch.tensor([0, size, size + 1, size + 2]))
    assert components._packed_sizes is components._element_shapes is components._cached_storage is None


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cpu_contraction_handles_sparse_ids_duplicates_and_self_loops(graph_type, dtype):
    edges = torch.tensor([[-7, 100], [100, -7], [100, 5], [100, 5], [5, 5], [900, 900]], dtype=dtype)
    graph = graph_type(edge_index=edges, nodes=edges.new_tensor([-9999, 12345]))

    components = graph.connected_components()

    assert components.concat.dtype == dtype
    assert components.concat.device.type == "cpu"
    assert components.tolist() == [[-9999], [-7, 5, 100], [900], [12345]]


@pytest.mark.parametrize("num_nodes", [0, 1, 11])
def test_cpu_contraction_handles_edgeless_dense_graphs(num_nodes):
    edges = torch.empty((0, 2), dtype=torch.long)

    labels = _component_labels_cpu(edges, num_nodes)

    assert torch.equal(labels, torch.arange(num_nodes))


@pytest.mark.parametrize("graph_type", [UndirectedGraph, DirectedGraph])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seed", [0, 7, 23])
def test_cpu_contraction_random_partitions_match_scipy(graph_type, dtype, seed):
    import numpy as np

    sparse = pytest.importorskip("scipy.sparse")
    csgraph = pytest.importorskip("scipy.sparse.csgraph")
    generator = torch.Generator().manual_seed(seed)
    for num_nodes in (0, 1, 7, 31, 79):
        edges = (
            torch.randint(num_nodes, (3 * num_nodes, 2), generator=generator, dtype=dtype)
            if num_nodes
            else torch.empty((0, 2), dtype=dtype)
        )
        endpoints = edges.numpy()
        matrix = sparse.coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (endpoints[:, 0], endpoints[:, 1])),
            shape=(num_nodes, num_nodes),
        )
        count, reference = csgraph.connected_components(matrix, directed=False)
        first = np.full(num_nodes, num_nodes, dtype=np.int64)
        np.minimum.at(first, reference, np.arange(num_nodes))
        expected = torch.from_numpy(first[reference])
        graph = graph_type(edge_index=edges, num_nodes=num_nodes)

        components = graph.connected_components()

        assert len(components) == count
        assert torch.equal(_labels_from_components(components, num_nodes), expected)


def test_cpu_contraction_does_not_export_graph_data(monkeypatch):
    graph = UndirectedGraph(edge_index=torch.tensor([[4, -7], [-7, 11], [11, 11]]), nodes=torch.tensor([-100]))

    def reject_export(*args, **kwargs):
        raise AssertionError("CPU component traversal must operate on tensors")

    with monkeypatch.context() as patch:
        for name in ("tolist", "cpu", "numpy"):
            patch.setattr(torch.Tensor, name, reject_export)
        components = graph.connected_components()

    assert components.tolist() == [[-100], [-7, 4, 11]]
