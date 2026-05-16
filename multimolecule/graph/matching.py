# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
#
# This file is part of MultiMolecule.
#
# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.
#
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import overload

import numpy as np
import torch
from torch import Tensor

Edge = tuple[int, int]


@overload
def maximum_weight_matching(
    edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int | None = None
) -> np.ndarray: ...


@overload
def maximum_weight_matching(  # type: ignore[overload-cannot-match]
    edge_index: Tensor,
    edge_weight: Tensor | np.ndarray,
    num_nodes: int | None = None,
) -> Tensor: ...


def maximum_weight_matching(
    edge_index: Tensor | np.ndarray, edge_weight: Tensor | np.ndarray, num_nodes: int | None = None
) -> Tensor | np.ndarray:
    """
    Compute an exact maximum-weight matching on an undirected simple graph.

    Args:
        edge_index: ``(E, 2)`` integer edge endpoints.
        edge_weight: ``(E,)`` edge weights. Non-positive and non-finite weights are ignored.
        num_nodes: Optional number of nodes. Isolated nodes are irrelevant and are not materialized.

    Returns:
        Matched edges with the same array backend as ``edge_index``.

    Notes:
        This is an in-house indexed Edmonds blossom/primal-dual implementation, specialized to the
        max-weight objective used by RNA contact-map decoding. It intentionally does not implement a
        max-cardinality tie-breaking objective.
    """
    if isinstance(edge_index, Tensor):
        device = edge_index.device
        edge_index_np = edge_index.detach().cpu().numpy()
        edge_weight_np = (
            edge_weight.detach().cpu().numpy() if isinstance(edge_weight, Tensor) else np.asarray(edge_weight)
        )
        out = _numpy_maximum_weight_matching(edge_index_np, edge_weight_np, num_nodes=num_nodes)
        return torch.as_tensor(out, dtype=torch.long, device=device)
    if isinstance(edge_index, np.ndarray):
        edge_weight_np = (
            edge_weight.detach().cpu().numpy() if isinstance(edge_weight, Tensor) else np.asarray(edge_weight)
        )
        return _numpy_maximum_weight_matching(edge_index, edge_weight_np, num_nodes=num_nodes)
    raise TypeError("edge_index must be a torch.Tensor or numpy.ndarray")


def _numpy_maximum_weight_matching(
    edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int | None = None
) -> np.ndarray:
    edge_index = np.asarray(edge_index)
    edge_weight = np.asarray(edge_weight)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"edge_index must have shape (E, 2), but got {edge_index.shape}.")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[0]:
        raise ValueError("edge_weight must be a 1D array aligned with edge_index.")
    if not np.issubdtype(edge_index.dtype, np.integer):
        raise TypeError("edge_index must contain integer endpoints.")
    if not (np.issubdtype(edge_weight.dtype, np.integer) or np.issubdtype(edge_weight.dtype, np.floating)):
        raise TypeError("edge_weight must contain real numeric values.")
    if num_nodes is not None:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
            raise TypeError("num_nodes must be an integer.")
        if num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, but got {num_nodes}.")
        num_nodes = int(num_nodes)
    if edge_index.shape[0] == 0:
        return np.empty((0, 2), dtype=int)
    if np.any(edge_index < 0):
        raise ValueError("edge_index must contain non-negative endpoints.")

    edge_weights: dict[Edge, float] = {}
    integer_weights = True
    for raw_edge, raw_weight in zip(edge_index, edge_weight):
        u = int(raw_edge[0])
        v = int(raw_edge[1])
        if u == v:
            continue
        if num_nodes is not None and not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge ({u}, {v}) is out of bounds for num_nodes={num_nodes}.")
        weight = raw_weight.item() if isinstance(raw_weight, np.generic) else raw_weight
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("edge_weight must contain real numeric values.")
        if not np.isfinite(weight) or weight <= 0:
            continue
        a, b = (u, v) if u < v else (v, u)
        previous = edge_weights.get((a, b))
        if previous is None or weight > previous:
            edge_weights[(a, b)] = weight
        integer_weights = integer_weights and isinstance(weight, Integral)

    if not edge_weights:
        return np.empty((0, 2), dtype=int)

    matched = _maximum_weight_matching_components(edge_weights, integer_weights=integer_weights)
    if not matched:
        return np.empty((0, 2), dtype=int)
    out = np.array(sorted(matched), dtype=int).reshape(-1, 2)
    return out


def _maximum_weight_matching_components(edge_weights: dict[Edge, float], *, integer_weights: bool) -> list[Edge]:
    neighbors: dict[int, list[int]] = defaultdict(list)
    degree: dict[int, int] = defaultdict(int)
    for u, v in edge_weights:
        neighbors[u].append(v)
        neighbors[v].append(u)
        degree[u] += 1
        degree[v] += 1
    if all(count <= 1 for count in degree.values()):
        return list(edge_weights)

    matched: list[Edge] = []
    seen: set[int] = set()
    for start in sorted(neighbors):
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbor for neighbor in neighbors[node] if neighbor not in component)
        seen.update(component)
        component_edges = {edge: weight for edge, weight in edge_weights.items() if edge[0] in component}
        if len(component_edges) == 1:
            matched.extend(component_edges)
        else:
            matched.extend(_edmonds_maximum_weight_matching(component_edges, integer_weights=integer_weights))
    return matched


@dataclass(eq=False, slots=True)
class _Blossom:
    children: list[int | _Blossom] = field(default_factory=list)
    edges: list[Edge | None] = field(default_factory=list)
    best_edges: list[Edge] | None = None

    def leaves(self):
        stack = [*self.children]
        while stack:
            item = stack.pop()
            if isinstance(item, _Blossom):
                stack.extend(item.children)
            else:
                yield item


_NO_NODE = object()
_LABEL_OUTER = 1
_LABEL_INNER = 2
_LABEL_SEEN = 4
_LABEL_SCANNED_OUTER = _LABEL_OUTER | _LABEL_SEEN
_DELTA_DONE = 1
_DELTA_UNLABELED_VERTEX = 2
_DELTA_OUTER_BLOSSOM = 3
_DELTA_INNER_BLOSSOM = 4


def _edmonds_maximum_weight_matching(edge_weights: dict[Edge, float], *, integer_weights: bool) -> list[Edge]:
    # The inline asserts document invariants and may be stripped under python -O.
    # For integer weights, verify_optimum() is the load-bearing post-condition check.
    nodes = sorted({node for edge in edge_weights for node in edge})
    if not nodes:
        return []
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    weights: list[dict[int, float]] = [{} for _ in range(n)]
    output_edge: dict[Edge, Edge] = {}
    max_weight = 0.0
    for (u_orig, v_orig), weight in edge_weights.items():
        u = node_to_idx[u_orig]
        v = node_to_idx[v_orig]
        neighbors[u].append(v)
        neighbors[v].append(u)
        weights[u][v] = weight
        weights[v][u] = weight
        output_edge[(u, v) if u < v else (v, u)] = (u_orig, v_orig) if u_orig < v_orig else (v_orig, u_orig)
        max_weight = max(max_weight, weight)

    mate = [-1] * n
    label: dict[int | _Blossom, int | None] = {}
    label_edge: dict[int | _Blossom, Edge | None] = {}
    in_blossom: list[int | _Blossom] = list(range(n))
    blossom_parent: dict[int | _Blossom, int | _Blossom | None] = dict.fromkeys(range(n))
    blossom_base: dict[int | _Blossom, int] = {node: node for node in range(n)}
    best_edge: dict[int | _Blossom, Edge | None] = {}
    dual_var = [max_weight] * n
    blossom_dual: dict[_Blossom, float] = {}
    allowed_edge: set[Edge] = set()
    queue: list[int] = []

    def canonical(u: int, v: int) -> Edge:
        return (u, v) if u < v else (v, u)

    def slack(u: int, v: int) -> float:
        return dual_var[u] + dual_var[v] - 2 * weights[u][v]

    def assign_label(node: int, label_value: int, through: int | None) -> None:
        blossom = in_blossom[node]
        assert label.get(node) is None and label.get(blossom) is None
        label[node] = label[blossom] = label_value
        label_edge[node] = label_edge[blossom] = None if through is None else (through, node)
        best_edge[node] = best_edge[blossom] = None
        if label_value == _LABEL_OUTER:
            if isinstance(blossom, _Blossom):
                queue.extend(blossom.leaves())
            else:
                queue.append(blossom)
        elif label_value == _LABEL_INNER:
            base = blossom_base[blossom]
            assign_label(mate[base], _LABEL_OUTER, base)

    def scan_blossom(v: int, w: int):
        path: list[int | _Blossom] = []
        base: int | object = _NO_NODE
        while v is not _NO_NODE:
            blossom = in_blossom[v]
            if label[blossom] & _LABEL_SEEN:  # type: ignore[operator]
                base = blossom_base[blossom]
                break
            assert label[blossom] == _LABEL_OUTER
            path.append(blossom)
            label[blossom] = _LABEL_SCANNED_OUTER
            edge = label_edge[blossom]
            if edge is None:
                assert mate[blossom_base[blossom]] == -1
                v = _NO_NODE  # type: ignore[assignment]
            else:
                assert edge[0] == mate[blossom_base[blossom]]
                v = edge[0]
                blossom = in_blossom[v]
                assert label[blossom] == _LABEL_INNER
                v = label_edge[blossom][0]  # type: ignore[index]
            if w is not _NO_NODE:
                v, w = w, v
        for blossom in path:
            label[blossom] = _LABEL_OUTER
        return base

    def add_blossom(base: int, v: int, w: int) -> None:
        base_blossom = in_blossom[base]
        v_blossom = in_blossom[v]
        w_blossom = in_blossom[w]
        blossom = _Blossom()
        blossom_base[blossom] = base
        blossom_parent[blossom] = None
        blossom_parent[base_blossom] = blossom
        path = blossom.children
        edges = blossom.edges
        edges.append((v, w))

        while v_blossom != base_blossom:
            blossom_parent[v_blossom] = blossom
            path.append(v_blossom)
            edge = label_edge[v_blossom]
            assert edge is not None
            edges.append(edge)
            v = label_edge[v_blossom][0]  # type: ignore[index]
            v_blossom = in_blossom[v]
        path.append(base_blossom)
        path.reverse()
        edges.reverse()

        while w_blossom != base_blossom:
            blossom_parent[w_blossom] = blossom
            path.append(w_blossom)
            edge = label_edge[w_blossom]
            edges.append((edge[1], edge[0]))  # type: ignore[index]
            w = edge[0]  # type: ignore[index]
            w_blossom = in_blossom[w]

        label[blossom] = _LABEL_OUTER
        label_edge[blossom] = label_edge[base_blossom]
        blossom_dual[blossom] = 0
        for leaf in blossom.leaves():
            if label[in_blossom[leaf]] == _LABEL_INNER:
                queue.append(leaf)
            in_blossom[leaf] = blossom

        best_to: dict[int | _Blossom, Edge] = {}
        for child in path:
            if isinstance(child, _Blossom):
                if child.best_edges is not None:
                    candidate_edges = child.best_edges
                    child.best_edges = None
                else:
                    candidate_edges = [(x, y) for x in child.leaves() for y in neighbors[x]]
            else:
                candidate_edges = [(child, y) for y in neighbors[child]]
            for edge in candidate_edges:
                i, j = edge
                if in_blossom[j] == blossom:
                    i, j = j, i
                other = in_blossom[j]
                if other != blossom and label.get(other) == _LABEL_OUTER:
                    current = best_to.get(other)
                    if current is None or slack(i, j) < slack(*current):
                        best_to[other] = edge
            best_edge[child] = None

        blossom.best_edges = list(best_to.values())
        best_edge[blossom] = None
        for edge in blossom.best_edges:
            current = best_edge[blossom]
            if current is None or slack(*edge) < slack(*current):
                best_edge[blossom] = edge

    def expand_blossom(blossom: _Blossom, end_stage: bool) -> None:
        def recurse(current: _Blossom, end: bool):
            for child in current.children:
                blossom_parent[child] = None
                if isinstance(child, _Blossom):
                    if end and blossom_dual[child] == 0:
                        yield child
                    else:
                        for leaf in child.leaves():
                            in_blossom[leaf] = child
                else:
                    in_blossom[child] = child

            if (not end) and label.get(current) == _LABEL_INNER:
                entry_child = in_blossom[label_edge[current][1]]  # type: ignore[index]
                idx = current.children.index(entry_child)
                step = 1 if idx & 1 else -1
                if idx & 1:
                    idx -= len(current.children)
                v, w = label_edge[current]  # type: ignore[misc]
                while idx != 0:
                    edge = current.edges[idx] if step == 1 else current.edges[idx - 1]
                    assert edge is not None
                    p, q = edge if step == 1 else (edge[1], edge[0])
                    label[w] = None
                    label[q] = None
                    assign_label(w, _LABEL_INNER, v)
                    allowed_edge.add((p, q))
                    allowed_edge.add((q, p))
                    idx += step
                    edge = current.edges[idx] if step == 1 else current.edges[idx - 1]
                    assert edge is not None
                    v, w = edge if step == 1 else (edge[1], edge[0])
                    allowed_edge.add((v, w))
                    allowed_edge.add((w, v))
                    idx += step
                next_child = current.children[idx]
                label[w] = label[next_child] = _LABEL_INNER
                label_edge[w] = label_edge[next_child] = (v, w)
                best_edge[next_child] = None
                idx += step
                while current.children[idx] != entry_child:
                    child = current.children[idx]
                    if label.get(child) == _LABEL_OUTER:
                        idx += step
                        continue
                    if isinstance(child, _Blossom):
                        for x in child.leaves():
                            if label.get(x):
                                break
                    else:
                        x = child
                    if label.get(x):
                        label[x] = None
                        label[mate[blossom_base[child]]] = None
                        assign_label(x, _LABEL_INNER, label_edge[x][0])  # type: ignore[index]
                    idx += step

            label.pop(current, None)
            label_edge.pop(current, None)
            best_edge.pop(current, None)
            del blossom_parent[current]
            del blossom_base[current]
            del blossom_dual[current]

        stack = [recurse(blossom, end_stage)]
        while stack:
            top = stack[-1]
            for child in top:
                stack.append(recurse(child, end_stage))
                break
            else:
                stack.pop()

    def augment_blossom(blossom: _Blossom, node: int) -> None:
        def recurse(current: _Blossom, x: int):
            child: int | _Blossom = x
            while blossom_parent[child] != current:
                child = blossom_parent[child]  # type: ignore[assignment]
            if isinstance(child, _Blossom):
                yield child, x
            start_idx = current.children.index(child)
            idx = start_idx
            step = 1 if start_idx & 1 else -1
            if start_idx & 1:
                idx -= len(current.children)
            while idx != 0:
                idx += step
                child = current.children[idx]
                edge = current.edges[idx] if step == 1 else current.edges[idx - 1]
                assert edge is not None
                w, y = edge if step == 1 else (edge[1], edge[0])
                if isinstance(child, _Blossom):
                    yield child, w
                idx += step
                child = current.children[idx]
                if isinstance(child, _Blossom):
                    yield child, y
                mate[w] = y
                mate[y] = w
            current.children = current.children[start_idx:] + current.children[:start_idx]
            current.edges = current.edges[start_idx:] + current.edges[:start_idx]
            blossom_base[current] = blossom_base[current.children[0]]
            assert blossom_base[current] == x

        stack = [recurse(blossom, node)]
        while stack:
            top = stack[-1]
            for args in top:
                stack.append(recurse(*args))
                break
            else:
                stack.pop()

    def augment_matching(v: int, w: int) -> None:
        for start, target in ((v, w), (w, v)):
            node = start
            target_node = target
            while True:
                blossom = in_blossom[node]
                assert label[blossom] == _LABEL_OUTER
                if isinstance(blossom, _Blossom):
                    augment_blossom(blossom, node)
                mate[node] = target_node
                edge = label_edge[blossom]
                if edge is None:
                    break
                matched = edge[0]
                other = in_blossom[matched]
                node, target_node = label_edge[other]  # type: ignore[misc]
                if isinstance(other, _Blossom):
                    augment_blossom(other, target_node)
                mate[target_node] = node

    def verify_optimum() -> None:
        assert min(dual_var) >= 0
        assert len(blossom_dual) == 0 or min(blossom_dual.values()) >= 0
        for i, j in output_edge:
            s = dual_var[i] + dual_var[j] - 2 * weights[i][j]
            i_blossoms: list[int | _Blossom] = [i]
            j_blossoms: list[int | _Blossom] = [j]
            while blossom_parent[i_blossoms[-1]] is not None:
                i_blossoms.append(blossom_parent[i_blossoms[-1]])  # type: ignore[arg-type]
            while blossom_parent[j_blossoms[-1]] is not None:
                j_blossoms.append(blossom_parent[j_blossoms[-1]])  # type: ignore[arg-type]
            i_blossoms.reverse()
            j_blossoms.reverse()
            for bi, bj in zip(i_blossoms, j_blossoms):
                if bi != bj:
                    break
                s += 2 * blossom_dual[bi]  # type: ignore[index]
            assert s >= 0
            if mate[i] == j or mate[j] == i:
                assert mate[i] == j and mate[j] == i
                assert s == 0
        for node in range(n):
            assert mate[node] != -1 or dual_var[node] == 0
        for blossom, dual in blossom_dual.items():
            if dual > 0:
                for edge in blossom.edges[1::2]:
                    assert edge is not None
                    i, j = edge
                    assert mate[i] == j and mate[j] == i

    while True:
        label.clear()
        label_edge.clear()
        best_edge.clear()
        for blossom in blossom_dual:
            blossom.best_edges = None
        allowed_edge.clear()
        queue[:] = []

        for node in range(n):
            if mate[node] == -1 and label.get(in_blossom[node]) is None:
                assign_label(node, _LABEL_OUTER, None)

        augmented = False
        while True:
            while queue and not augmented:
                v = queue.pop()
                assert label[in_blossom[v]] == _LABEL_OUTER
                for w in neighbors[v]:
                    v_blossom = in_blossom[v]
                    w_blossom = in_blossom[w]
                    if v_blossom == w_blossom:
                        continue
                    edge_slack = slack(v, w)
                    if (v, w) not in allowed_edge and edge_slack <= 0:
                        allowed_edge.add((v, w))
                        allowed_edge.add((w, v))
                    if (v, w) in allowed_edge:
                        if label.get(w_blossom) is None:
                            assign_label(w, _LABEL_INNER, v)
                        elif label.get(w_blossom) == _LABEL_OUTER:
                            base = scan_blossom(v, w)
                            if base is not _NO_NODE:
                                add_blossom(base, v, w)  # type: ignore[arg-type]
                            else:
                                augment_matching(v, w)
                                augmented = True
                                break
                        elif label.get(w) is None:
                            label[w] = _LABEL_INNER
                            label_edge[w] = (v, w)
                    elif label.get(w_blossom) == _LABEL_OUTER:
                        edge = best_edge.get(v_blossom)
                        if edge is None or edge_slack < slack(*edge):
                            best_edge[v_blossom] = (v, w)
                    elif label.get(w) is None:
                        edge = best_edge.get(w)
                        if edge is None or edge_slack < slack(*edge):
                            best_edge[w] = (v, w)

            if augmented:
                break

            delta_type = _DELTA_DONE
            delta = min(dual_var)
            delta_edge: Edge | None = None
            delta_blossom: _Blossom | None = None

            for node in range(n):
                edge = best_edge.get(node)
                if label.get(in_blossom[node]) is None and edge is not None:
                    candidate = slack(*edge)
                    if candidate < delta:
                        delta = candidate
                        delta_type = _DELTA_UNLABELED_VERTEX
                        delta_edge = edge

            for item, parent in blossom_parent.items():
                edge = best_edge.get(item)
                if parent is None and label.get(item) == _LABEL_OUTER and edge is not None:
                    edge_slack = slack(*edge)
                    candidate = edge_slack // 2 if integer_weights else edge_slack / 2.0
                    if candidate < delta:
                        delta = candidate
                        delta_type = _DELTA_OUTER_BLOSSOM
                        delta_edge = edge

            for blossom, dual in blossom_dual.items():
                if blossom_parent[blossom] is None and label.get(blossom) == _LABEL_INNER and dual < delta:
                    delta = dual
                    delta_type = _DELTA_INNER_BLOSSOM
                    delta_blossom = blossom

            for node in range(n):
                if label.get(in_blossom[node]) == _LABEL_OUTER:
                    dual_var[node] -= delta
                elif label.get(in_blossom[node]) == _LABEL_INNER:
                    dual_var[node] += delta
            for blossom in blossom_dual:
                if blossom_parent[blossom] is None:
                    if label.get(blossom) == _LABEL_OUTER:
                        blossom_dual[blossom] += delta
                    elif label.get(blossom) == _LABEL_INNER:
                        blossom_dual[blossom] -= delta

            if delta_type == _DELTA_DONE:
                break
            if delta_type in (_DELTA_UNLABELED_VERTEX, _DELTA_OUTER_BLOSSOM):
                assert delta_edge is not None
                v, w = delta_edge
                allowed_edge.add((v, w))
                allowed_edge.add((w, v))
                queue.append(v)
            elif delta_type == _DELTA_INNER_BLOSSOM:
                assert delta_blossom is not None
                expand_blossom(delta_blossom, False)

        for node, matched in enumerate(mate):
            if matched != -1:
                assert mate[matched] == node
        if not augmented:
            break
        for blossom in list(blossom_dual.keys()):
            should_expand = all(
                (
                    blossom in blossom_dual,
                    blossom_parent[blossom] is None,
                    label.get(blossom) == _LABEL_OUTER,
                    blossom_dual[blossom] == 0,
                )
            )
            if should_expand:
                expand_blossom(blossom, True)

    if integer_weights:
        verify_optimum()

    return [
        output_edge[canonical(node, matched)] for node, matched in enumerate(mate) if matched != -1 and node < matched
    ]
