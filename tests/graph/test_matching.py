# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import itertools
import inspect
import math
from random import Random

import pytest
import torch

from multimolecule.graph import maximum_weight_matching

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _score(edges, weights, selected):
    by_edge = {}
    for (u, v), weight in zip(edges, weights):
        edge = min(u, v), max(u, v)
        by_edge[edge] = max(by_edge.get(edge, 0), weight)
    return sum(by_edge[tuple(edge)] for edge in selected)


def _brute_force_score(edges, weights):
    best = 0
    for keep in itertools.product((False, True), repeat=len(edges)):
        selected = [edge for edge, take in zip(edges, keep) if take]
        vertices = [vertex for edge in selected for vertex in edge]
        if len(vertices) == len(set(vertices)):
            best = max(best, sum(weight for weight, take in zip(weights, keep) if take))
    return best


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_maximum_weight_matching_prefers_global_optimum(device, dtype):
    edges = torch.tensor([(0, 1), (0, 2), (1, 3)], dtype=dtype, device=device)
    weights = torch.tensor([0.9, 0.8, 0.8], device=device)

    got = maximum_weight_matching(edges, weights)

    assert got.dtype == edges.dtype
    assert got.device == edges.device
    assert got.tolist() == [[0, 2], [1, 3]]


@pytest.mark.parametrize("device", DEVICES)
def test_maximum_weight_matching_handles_disconnected_sparse_nodes(device):
    edges = torch.tensor([(10**9, 10**9 + 1), (0, 1), (8, 9), (0, 2), (1, 3)], device=device)
    weights = torch.tensor([0.4, 0.9, 0.6, 0.8, 0.8], device=device)

    got = maximum_weight_matching(edges, weights)

    assert got.tolist() == [[0, 2], [1, 3], [8, 9], [10**9, 10**9 + 1]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("include_single_edges", [False, True])
def test_matching_bounds_blossom_tables_by_component(monkeypatch, device, include_single_edges):
    import multimolecule.graph.matching as matching_module

    sizes = []
    original = matching_module._TensorBlossom.__init__

    def record(self, edges, weights, num_nodes):
        sizes.append(num_nodes)
        original(self, edges, weights, num_nodes)

    monkeypatch.setattr(matching_module._TensorBlossom, "__init__", record)
    edges, weights, expected = [], [], []
    for component in range(12):
        base = 1000 * component
        edges.extend([(base, base + 1), (base, base + 2), (base + 1, base + 2)])
        weights.extend([9, 8, 7])
        expected.append([base, base + 1])
        if include_single_edges:
            edges.append((base + 5, base + 6))
            weights.append(6)
            expected.append([base + 5, base + 6])

    got = maximum_weight_matching(torch.tensor(edges, device=device), torch.tensor(weights, device=device))

    assert got.tolist() == expected
    assert sizes == [3] * 12


@pytest.mark.parametrize("device", DEVICES)
def test_matching_preserves_large_integer_weight_differences(device):
    base = 2**55
    edges = torch.tensor([(0, 1), (0, 2), (1, 3)], device=device)
    weights = torch.tensor([2 * base + 1, base, base], device=device)

    got = maximum_weight_matching(edges, weights)

    assert got.tolist() == [[0, 1]]


@pytest.mark.parametrize("device", DEVICES)
def test_matching_rescales_extreme_floats_without_losing_the_optimum(device):
    edges = torch.tensor([(0, 1), (0, 2), (1, 3)], device=device)
    weights = torch.tensor([1e308, 8e307, 8e307], dtype=torch.float64, device=device)

    got = maximum_weight_matching(edges, weights)

    assert got.tolist() == [[0, 2], [1, 3]]


@pytest.mark.parametrize("dtype,bound", [(torch.int8, 128), (torch.uint8, 256), (torch.int64, 2**64)])
def test_matching_node_count_above_endpoint_dtype_range(dtype, bound):
    edges = torch.tensor([(0, 1)], dtype=dtype)

    got = maximum_weight_matching(edges, torch.tensor([1.0]), num_nodes=bound)

    assert torch.equal(got, edges)


@pytest.mark.parametrize("device", DEVICES)
def test_maximum_weight_matching_random_small_against_bruteforce(device):
    rng = Random(0)
    all_edges = list(itertools.combinations(range(7), 2))
    for _ in range(20):
        edges = rng.sample(all_edges, 8)
        weights = [rng.uniform(0.1, 1.0) for _ in edges]

        got = maximum_weight_matching(
            torch.tensor(edges, device=device), torch.tensor(weights, dtype=torch.float64, device=device)
        )

        assert math.isclose(_score(edges, weights, got.tolist()), _brute_force_score(edges, weights))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("integer_weights", [False, True])
def test_maximum_weight_matching_random_against_networkx(device, integer_weights):
    nx = pytest.importorskip("networkx")
    rng = Random(1)
    for n, p in [(20, 0.1), (20, 0.4), (40, 0.05), (40, 0.2)]:
        for _ in range(5):
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
            weights = [rng.randrange(1, 30) if integer_weights else rng.uniform(0.1, 1) for _ in edges]
            graph = nx.Graph()
            graph.add_weighted_edges_from((u, v, weight) for (u, v), weight in zip(edges, weights))
            expected = sorted((min(u, v), max(u, v)) for u, v in nx.max_weight_matching(graph))

            got = maximum_weight_matching(
                torch.tensor(edges, device=device).reshape(-1, 2),
                torch.tensor(weights, dtype=torch.int64 if integer_weights else torch.float64, device=device),
            )

            actual = got.tolist()
            assert len({node for edge in actual for node in edge}) == got.numel()
            assert math.isclose(_score(edges, weights, actual), _score(edges, weights, expected))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("integer_weights", [False, True])
def test_nested_blossom_contraction_expansion_and_augmentation(device, integer_weights):
    weighted_edges = [
        (1, 2, 8),
        (1, 3, 8),
        (2, 3, 10),
        (2, 4, 12),
        (3, 5, 12),
        (4, 5, 14),
        (4, 6, 12),
        (5, 7, 12),
        (6, 7, 14),
        (7, 8, 12),
    ]
    edges = torch.tensor([(u, v) for u, v, _ in weighted_edges], device=device)
    weights = torch.tensor(
        [w for _, _, w in weighted_edges], dtype=torch.int64 if integer_weights else torch.float64, device=device
    )

    got = maximum_weight_matching(edges, weights)

    assert got.tolist() == [[1, 2], [3, 5], [4, 6], [7, 8]]


@pytest.mark.parametrize("device", DEVICES)
def test_maximum_weight_matching_ignores_invalid_weights_and_uses_maximum_duplicate(device):
    edges = torch.tensor([(0, 1), (1, 0), (0, 2), (1, 3), (5, 5), (6, 7), (8, 9), (10, 11)], device=device)
    weights = torch.tensor([0.1, 0.9, 0.8, 0.8, 100, float("nan"), float("inf"), -1], device=device)

    got = maximum_weight_matching(edges, weights)

    assert got.tolist() == [[0, 2], [1, 3]]


@pytest.mark.parametrize("device", DEVICES)
def test_matching_does_not_export_tensors_to_host(monkeypatch, device):
    edges = torch.tensor([(0, 1), (1, 2), (2, 0), (2, 3)], device=device)
    weights = torch.tensor([5, 6, 5, 4], device=device)

    def reject(*args, **kwargs):
        raise AssertionError("Matching must not export tensor data to a CPU backend")

    original_tolist = torch.Tensor.tolist

    def metadata_only(tensor):
        frame = inspect.currentframe().f_back
        danling_metadata = (
            tensor.device.type == "cpu" and "danling/tensors/nested_tensor.py" in frame.f_code.co_filename
        )
        component_sizes = frame.f_code.co_name == "_connected_components" and tensor.ndim == 1
        if not (danling_metadata or component_sizes) or tensor.data_ptr() in (edges.data_ptr(), weights.data_ptr()):
            reject()
        return original_tolist(tensor)

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy"):
            patch.setattr(torch.Tensor, name, reject)
        patch.setattr(torch.Tensor, "tolist", metadata_only)
        got = maximum_weight_matching(edges, weights)

    assert got.tolist() == [[0, 1], [2, 3]]
    assert got.device == edges.device


@pytest.mark.parametrize("device", DEVICES)
def test_empty_matching_preserves_device_and_dtype(device):
    edges = torch.empty((0, 2), dtype=torch.int32, device=device)
    got = maximum_weight_matching(edges, torch.empty(0, device=device))
    assert got.shape == (0, 2)
    assert got.dtype == edges.dtype and got.device == edges.device


def test_maximum_weight_matching_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="torch tensors"):
        maximum_weight_matching([[0, 1]], torch.tensor([1.0]))
    with pytest.raises(TypeError, match="integer"):
        maximum_weight_matching(torch.tensor([[0.9, 1.1]]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        maximum_weight_matching(torch.tensor([[-1, 2]]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="out of bounds"):
        maximum_weight_matching(torch.tensor([[0, 2]]), torch.tensor([1.0]), num_nodes=2)
    with pytest.raises(ValueError, match="aligned"):
        maximum_weight_matching(torch.tensor([[0, 2]]), torch.tensor([1.0, 2.0]))
    with pytest.raises(TypeError, match="real numeric"):
        maximum_weight_matching(torch.tensor([[0, 2]]), torch.tensor([True]))
    with pytest.raises(ValueError, match="arithmetic range"):
        maximum_weight_matching(torch.tensor([[0, 2], [0, 3]]), torch.tensor([torch.iinfo(torch.int64).max, 1]))
