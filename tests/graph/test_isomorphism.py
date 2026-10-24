# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

from collections import Counter
from itertools import permutations
from random import Random

import pytest
import torch

from multimolecule.graph import (
    DirectedGraph,
    ResourceLimitError,
    UndirectedGraph,
    automorphisms,
    is_isomorphic,
    isomorphisms,
)


def _graph(edges=(), *, nodes=None, num_nodes=None, directed=False, dtype=torch.int64, device="cpu"):
    graph_type = DirectedGraph if directed else UndirectedGraph
    return graph_type(
        torch.tensor(edges, dtype=dtype, device=device).reshape(-1, 2),
        nodes=None if nodes is None else torch.tensor(nodes, dtype=dtype, device=device),
        num_nodes=num_nodes,
        allow_multi_edges=True,
    )


def _rows(mappings):
    return {tuple(row) for row in mappings.tolist()}


def _brute_force(source, target, node_labels, other_node_labels, edge_labels, other_edge_labels):
    """Compare every bijection using labeled edge multisets."""
    source_nodes, target_nodes = source.nodes.tolist(), target.nodes.tolist()
    directed = isinstance(source, DirectedGraph)

    def edges(graph, labels, mapping):
        result = Counter()
        for (u, v), label in zip(graph.edge_index.tolist(), labels):
            u, v = mapping[u], mapping[v]
            if not directed and u > v:
                u, v = v, u
            result[u, v, label] += 1
        return result

    target_edges = edges(target, other_edge_labels, dict(zip(target_nodes, target_nodes)))
    target_colors = dict(zip(target_nodes, other_node_labels))
    return {
        row
        for row in permutations(target_nodes)
        if all(label == target_colors[node] for label, node in zip(node_labels, row))
        and edges(source, edge_labels, dict(zip(source_nodes, row))) == target_edges
    }


class TestMappings:
    def test_cycle_automorphisms(self, device):
        graph = _graph([(node, (node + 1) % 6) for node in range(6)], device=device)
        expected = {
            tuple((offset + direction * node) % 6 for node in range(6)) for offset in range(6) for direction in (-1, 1)
        }
        got = automorphisms(graph)
        assert got.shape == (12, 6)
        assert _rows(got) == expected
        assert got.dtype == graph.nodes.dtype and got.device == graph.device

    def test_fixed_nodes(self):
        graph = _graph([(0, 1), (0, 2), (0, 3)])
        assert _rows(automorphisms(graph)) == {(0, *leaves) for leaves in permutations((1, 2, 3))}
        assert _rows(automorphisms(graph, fixed_nodes=[1])) == {(0, 1, 2, 3), (0, 1, 3, 2)}

    def test_target_ids_layout_and_constraints(self, device):
        source = _graph([(-7, 100), (100, 999)], nodes=[-30], device=device)
        target = _graph([(5, 40), (40, 90)], nodes=[12], dtype=torch.int32, device=device)
        got = isomorphisms(source, target, fixed_mapping={-7: 90})
        assert got.tolist() == [[12, 90, 40, 5]]
        assert got.dtype == target.nodes.dtype and got.device == target.device
        assert not is_isomorphic(source, target, fixed_mapping={100: 5})

    def test_empty_graph(self):
        empty, isolated = _graph(), _graph(num_nodes=1)
        assert automorphisms(empty).shape == (1, 0)
        assert is_isomorphic(empty, empty)
        assert isomorphisms(empty, isolated).shape == (0, 0)
        assert isomorphisms(isolated, empty).shape == (0, 1)
        assert not is_isomorphic(empty, isolated)

    def test_directed_edges(self):
        outward = _graph([(0, 1), (0, 2)], directed=True)
        inward = _graph([(1, 0), (2, 0)], directed=True)
        assert _rows(automorphisms(outward)) == {(0, 1, 2), (0, 2, 1)}
        assert not is_isomorphic(outward, inward)
        assert not is_isomorphic(outward, _graph([(0, 1), (0, 2)]))

    @pytest.mark.parametrize("directed", [False, True])
    def test_loops_and_parallel_edges(self, directed):
        source = _graph([(0, 0), (0, 1), (0, 1)], directed=directed)
        target = _graph([(8, 7), (8, 8), (8, 7)], directed=directed)
        labels = {"edge_labels": ["loop", "a", "b"], "other_edge_labels": ["b", "loop", "a"]}
        assert isomorphisms(source, target, **labels).tolist() == [[8, 7]]
        assert not is_isomorphic(source, target, edge_labels=[1, 2, 2], other_edge_labels=[2, 1, 3])
        assert not is_isomorphic(source, _graph([(0, 0), (0, 1)], directed=directed))
        assert not is_isomorphic(source, _graph([(0, 1), (0, 1), (0, 1)], directed=directed))

    def test_regular_nonisomorphic_graphs(self):
        cycle = _graph([(node, (node + 1) % 6) for node in range(6)])
        triangles = _graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
        assert isomorphisms(cycle, triangles).shape == (0, 6)
        assert not is_isomorphic(cycle, triangles)

    def test_edge_table_mutation_is_observed(self):
        source = _graph([(0, 1), (1, 2)])
        target = _graph([(10, 20)])
        assert not is_isomorphic(source, target)
        target.edges.extend(torch.tensor([[20, 30]]))
        assert _rows(isomorphisms(source, target)) == {(10, 20, 30), (30, 20, 10)}

    @pytest.mark.parametrize("directed", [False, True])
    def test_small_labeled_multigraphs_against_brute_force(self, directed):
        rng = Random(7)
        for trial in range(4):
            possible = [(u, v) for u in range(4) for v in range(4) if directed or u <= v]
            edges = [edge for edge in possible if rng.random() < 0.3]
            edges += edges[:1]
            relabel = [10, 20, 30, 40]
            rng.shuffle(relabel)
            source = _graph(edges, num_nodes=4, directed=directed)
            target = _graph([(relabel[u], relabel[v]) for u, v in edges], nodes=relabel, directed=directed)
            labels = [rng.randrange(2) for _ in range(4)]
            target_labels = [labels[relabel.index(node)] for node in target.nodes.tolist()]
            if trial % 2:
                target_labels[0] += 1
            edge_labels = [rng.randrange(2) for _ in edges]
            options = {
                "node_labels": labels,
                "other_node_labels": target_labels,
                "edge_labels": edge_labels,
                "other_edge_labels": edge_labels,
            }
            expected = _brute_force(source, target, **options)
            assert _rows(isomorphisms(source, target, **options)) == expected
            assert is_isomorphic(source, target, **options) is bool(expected)


class TestLabels:
    @pytest.fixture
    def star(self):
        return _graph([(0, 1), (0, 2), (0, 3)])

    @pytest.mark.parametrize("labels", [{"node_labels": ["C", "O", "O", "N"]}, {"edge_labels": [1, 1, 2]}])
    def test_labels_break_symmetry(self, star, labels):
        assert _rows(automorphisms(star, **labels)) == {(0, 1, 2, 3), (0, 2, 1, 3)}

    def test_labels_are_compared_between_graphs(self):
        graph = _graph([(0, 1)])
        assert isomorphisms(graph, graph, node_labels=["C", "O"], other_node_labels=["O", "C"]).tolist() == [[1, 0]]
        assert not is_isomorphic(graph, graph, node_labels=["C", "O"], other_node_labels=["N", "S"])
        assert not is_isomorphic(graph, graph, edge_labels=["single"], other_edge_labels=["double"])

    def test_tensor_rows_and_scalar_items(self, device):
        graph = _graph([(0, 1), (0, 2), (0, 3)], device=device)
        rows = torch.tensor([[6, 0], [8, 0], [8, 0], [8, -1]], device=device)
        assert _rows(automorphisms(graph, node_labels=rows)) == {(0, 1, 2, 3), (0, 2, 1, 3)}
        scalars = rows[:, 0]
        expected = {(0, *leaves) for leaves in permutations((1, 2, 3))}
        assert _rows(automorphisms(graph, node_labels=scalars)) == expected
        assert _rows(automorphisms(graph, node_labels=scalars.tolist())) == expected
        with pytest.raises(TypeError, match="whole Tensor"):
            automorphisms(graph, node_labels=list(scalars))

    @pytest.mark.parametrize(
        ("operation", "name", "edges", "labels"),
        [
            (isomorphisms, "node_labels", [(0, 1)], [None, None]),
            (is_isomorphic, "other_node_labels", [], []),
            (isomorphisms, "edge_labels", [], []),
            (is_isomorphic, "other_edge_labels", [(0, 1)], [None]),
        ],
    )
    def test_label_arguments_must_be_paired(self, operation, name, edges, labels):
        graph = _graph(edges)
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            operation(graph, graph, **{name: labels})

    def test_hashable_python_labels(self, star):
        labels = [None, "oxygen", ("oxygen", -1), frozenset({8})]
        assert automorphisms(star, node_labels=labels).tolist() == [[0, 1, 2, 3]]
        assert is_isomorphic(star, star, edge_labels=[None] * 3, other_edge_labels=[None] * 3)

    @pytest.mark.parametrize("labels", [[("atom", torch.tensor(8))], [float("nan")]])
    def test_invalid_label_values(self, labels):
        with pytest.raises((TypeError, ValueError)):
            automorphisms(_graph(num_nodes=1), node_labels=labels)


class TestSearchLimits:
    def test_enumeration_limit_is_exact_and_never_truncates(self):
        graph = _graph(num_nodes=3)
        assert automorphisms(graph, max_mappings=6).shape == (6, 3)
        with pytest.raises(ResourceLimitError) as error:
            isomorphisms(graph, graph, max_mappings=5)
        assert error.value.resource == "max_mappings" and error.value.limit == 5

    @pytest.mark.parametrize("operation", [automorphisms, isomorphisms, is_isomorphic])
    def test_search_limit_raises_instead_of_returning_partial_results(self, operation):
        graph = _graph(num_nodes=4)
        args = (graph,) if operation is automorphisms else (graph, graph)
        options = {"mapping_filter": lambda _: False} if operation is is_isomorphic else {}
        with pytest.raises(ResourceLimitError) as error:
            operation(*args, max_states=1, **options)
        assert error.value.resource == "max_states" and error.value.limit == 1

    @pytest.mark.parametrize("accept", ["any", (2, 1, 0), None])
    def test_filter_acceptance_and_exhaustion(self, accept):
        graph = _graph(num_nodes=3)
        seen = []

        def predicate(mapping):
            seen.append(tuple(mapping.tolist()))
            return accept == "any" or seen[-1] == accept

        assert is_isomorphic(graph, graph, mapping_filter=predicate) is (accept is not None)
        assert len(seen) == len(set(seen))
        if accept == "any":
            assert len(seen) == 1
        elif accept is None:
            assert len(seen) == 6
        else:
            assert seen[-1] == accept

    def test_filter_exceptions_propagate(self):
        def fail(mapping):
            raise RuntimeError("validation failed")

        graph = _graph(num_nodes=1)
        with pytest.raises(RuntimeError, match="validation failed"):
            is_isomorphic(graph, graph, mapping_filter=fail)
