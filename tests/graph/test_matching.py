# MultiMolecule
# Copyright (C) 2024-Present MultiMolecule

import itertools
from random import Random

import pytest
import torch

from multimolecule.graph import maximum_weight_matching


@pytest.fixture(params=["tree", "odd_cycles", "general"])
def weighted_graph(request):
    if request.param == "tree":
        return [(7, 2, 3), (2, 9, 4), (2, 4, 8), (4, 1, 6), (7, 5, 2)]
    if request.param == "odd_cycles":
        edges = [
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
        return [(a, b, weight / 2) for a, b, weight in edges]
    random = Random(793)
    return [
        (a, b, 2**55 + random.randrange(-10, 11))
        for a, b in itertools.combinations(range(9), 2)
        if random.random() < 0.5
    ]


class TestMaximumWeightMatching:
    def test_objective_matches_networkx(self, device, weighted_graph):
        nx = pytest.importorskip("networkx")
        graph = nx.Graph()
        graph.add_weighted_edges_from(weighted_graph)
        expected = sum(graph[a][b]["weight"] for a, b in nx.max_weight_matching(graph))
        edges = torch.tensor([(a, b) for a, b, _ in weighted_graph], dtype=torch.int32, device=device)
        weights = torch.tensor([weight for _, _, weight in weighted_graph], device=device)

        pairs = maximum_weight_matching(edges, weights)

        selected = pairs.tolist()
        assert pairs.dtype == edges.dtype and pairs.device == edges.device
        assert pairs.unique().numel() == pairs.numel()
        assert all(a < b for a, b in selected)
        assert selected == sorted(selected)
        assert sum(graph[a][b]["weight"] for a, b in selected) == expected

    def test_disconnected_graph_with_sparse_node_ids(self, device):
        edges = torch.tensor([(10**9, 10**9 + 1), (0, 1), (8, 9), (0, 2), (1, 3)], device=device)
        weights = torch.tensor([0.4, 0.9, 0.6, 0.8, 0.8], device=device)

        pairs = maximum_weight_matching(edges, weights)

        assert pairs.tolist() == [[0, 2], [1, 3], [8, 9], [10**9, 10**9 + 1]]

    def test_duplicate_edges_and_ignored_weights(self, device):
        edges = torch.tensor([(0, 1), (1, 0), (1, 2), (5, 5), (6, 7), (8, 9), (10, 11), (12, 13)], device=device)
        weights = torch.tensor([1.0, 3.0, 2.0, 100, torch.nan, torch.inf, -1, 0], device=device)

        pairs = maximum_weight_matching(edges, weights)

        assert pairs.tolist() == [[0, 1]]

    def test_integer_differences_above_float64_precision(self, device):
        base = 2**55
        edges = torch.tensor([(0, 1), (0, 2), (1, 3)], device=device)
        weights = torch.tensor([2 * base + 1, base, base], device=device)

        pairs = maximum_weight_matching(edges, weights)

        assert pairs.tolist() == [[0, 1]]

    def test_integer_objective_larger_than_int64(self, device):
        size = 12
        edges = torch.stack((torch.arange(size - 1, device=device), torch.arange(1, size, device=device)), dim=1)
        weights = torch.full((size - 1,), torch.iinfo(torch.int64).max // 4, dtype=torch.int64, device=device)
        weights[::2] -= 1

        pairs = maximum_weight_matching(edges, weights)

        assert pairs.tolist() == [[node, node + 1] for node in range(0, size, 2)]
        assert sum(int(weights[a]) for a, _ in pairs) > torch.iinfo(torch.int64).max

    def test_extreme_floating_weights(self, device):
        edges = torch.tensor([(0, 1), (0, 2), (1, 3)], device=device)
        weights = torch.tensor([1e308, 8e307, 8e307], dtype=torch.float64, device=device)

        pairs = maximum_weight_matching(edges, weights)

        assert pairs.tolist() == [[0, 2], [1, 3]]

    def test_unrepresentable_dynamic_range_raises(self):
        edges = torch.tensor([[0, 1], [1, 2]])
        with pytest.raises(ValueError, match="dynamic range"):
            maximum_weight_matching(edges, torch.tensor([1e308, 5e-324], dtype=torch.float64))

    def test_empty_matching(self, device):
        edges = torch.empty((0, 2), dtype=torch.int32, device=device)

        pairs = maximum_weight_matching(edges, torch.empty(0, device=device))

        assert pairs.shape == (0, 2)
        assert pairs.dtype == edges.dtype and pairs.device == edges.device
