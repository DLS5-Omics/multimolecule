# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from multimolecule.graph import maximum_weight_matching


def _score(edge_index: np.ndarray, edge_weight: np.ndarray, selected: np.ndarray) -> float:
    weights = {tuple(edge): float(weight) for edge, weight in zip(edge_index.tolist(), edge_weight.tolist())}
    weights.update({(b, a): weight for (a, b), weight in list(weights.items())})
    return sum(weights[tuple(edge)] for edge in selected.tolist())


def _brute_force_matching(edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
    best_score = float("-inf")
    best_edges: list[tuple[int, int]] = []
    edges = [tuple(map(int, edge)) for edge in edge_index.tolist()]
    weights = [float(weight) for weight in edge_weight.tolist()]
    for keep in itertools.product((False, True), repeat=len(edges)):
        used: set[int] = set()
        selected: list[tuple[int, int]] = []
        score = 0.0
        valid = True
        for take, (a, b), weight in zip(keep, edges, weights):
            if not take:
                continue
            if a in used or b in used:
                valid = False
                break
            used.update((a, b))
            selected.append((min(a, b), max(a, b)))
            score += weight
        if valid and score > best_score:
            best_score = score
            best_edges = selected
    return np.array(sorted(best_edges), dtype=int).reshape(-1, 2)


def test_maximum_weight_matching_prefers_global_optimum() -> None:
    edge_index = np.array([(0, 1), (0, 2), (1, 3)])
    edge_weight = np.array([0.9, 0.8, 0.8])

    got = maximum_weight_matching(edge_index, edge_weight)

    assert got.tolist() == [[0, 2], [1, 3]]


def test_maximum_weight_matching_torch_backend() -> None:
    edge_index = torch.tensor([(0, 1), (0, 2), (1, 3)])
    edge_weight = torch.tensor([0.9, 0.8, 0.8])

    got = maximum_weight_matching(edge_index, edge_weight)

    assert isinstance(got, torch.Tensor)
    assert got.device == edge_index.device
    assert got.tolist() == [[0, 2], [1, 3]]


def test_maximum_weight_matching_random_small_against_bruteforce() -> None:
    rng = np.random.default_rng(0)
    all_edges = np.array(list(itertools.combinations(range(7), 2)), dtype=int)
    for _ in range(20):
        choice = rng.choice(len(all_edges), size=8, replace=False)
        edge_index = all_edges[choice]
        edge_weight = rng.uniform(0.1, 1.0, size=len(edge_index))

        got = maximum_weight_matching(edge_index, edge_weight)
        expected = _brute_force_matching(edge_index, edge_weight)

        assert np.isclose(_score(edge_index, edge_weight, got), _score(edge_index, edge_weight, expected))


def test_maximum_weight_matching_random_against_networkx() -> None:
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(1)
    for n, p in [(20, 0.1), (20, 0.4), (40, 0.05), (40, 0.2)]:
        for _ in range(5):
            edge_index = np.array(
                [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p],
                dtype=int,
            ).reshape(-1, 2)
            edge_weight = rng.uniform(0.1, 1.0, size=len(edge_index))
            graph = nx.Graph()
            for edge, weight in zip(edge_index.tolist(), edge_weight.tolist()):
                graph.add_edge(*edge, weight=weight)

            got = maximum_weight_matching(edge_index, edge_weight)
            expected = np.array(
                sorted((min(i, j), max(i, j)) for i, j in nx.max_weight_matching(graph, weight="weight")),
                dtype=int,
            ).reshape(-1, 2)

            assert np.isclose(_score(edge_index, edge_weight, got), _score(edge_index, edge_weight, expected))


def test_maximum_weight_matching_ignores_non_positive_edges() -> None:
    edge_index = np.array([(0, 1), (1, 2), (2, 3)])
    edge_weight = np.array([0.0, -1.0, 0.5])

    got = maximum_weight_matching(edge_index, edge_weight)

    assert got.tolist() == [[2, 3]]


def test_maximum_weight_matching_rejects_invalid_edge_index() -> None:
    with pytest.raises(TypeError, match="integer"):
        maximum_weight_matching(np.array([[0.9, 1.1]]), np.array([1.0]))
    with pytest.raises(TypeError, match="integer"):
        maximum_weight_matching(torch.tensor([[0.9, 1.1]]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        maximum_weight_matching(np.array([[-1, 2]]), np.array([1.0]))
    with pytest.raises(ValueError, match="out of bounds"):
        maximum_weight_matching(np.array([[0, 2]]), np.array([1.0]), num_nodes=2)
