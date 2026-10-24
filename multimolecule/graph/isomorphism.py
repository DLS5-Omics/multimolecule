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

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from numbers import Integral
from typing import TypeAlias

import torch
from torch import Tensor

from .topology import DirectedGraph, UndirectedGraph

_Graph: TypeAlias = DirectedGraph | UndirectedGraph
_Labels: TypeAlias = Sequence[Hashable] | Tensor | None
_Adjacency: TypeAlias = list[list[tuple[int, int]]]


class ResourceLimitError(RuntimeError):
    """An exact graph search exceeded a resource budget; its result is unknown."""

    def __init__(self, resource: str, limit: int) -> None:
        self.resource = resource
        self.limit = limit
        super().__init__(resource, limit)

    def __str__(self) -> str:
        return f"Graph isomorphism search exceeded {self.resource}={self.limit}."


def isomorphisms(
    graph: _Graph,
    other: _Graph,
    *,
    node_labels: _Labels = None,
    other_node_labels: _Labels = None,
    edge_labels: _Labels = None,
    other_edge_labels: _Labels = None,
    fixed_mapping: Mapping[int, int] | None = None,
    max_mappings: int = 10_000,
    max_states: int = 100_000,
) -> Tensor:
    """Enumerate every label-preserving bijection from ``graph`` to ``other``.

    Args:
        graph: Source directed or undirected graph, including isolated nodes.
        other: Target graph. Directed and undirected graphs are not isomorphic.
        node_labels: Hashable labels aligned with ``graph.nodes``. Pass a whole
            Tensor or a sequence of Python hashable values; Tensor items inside
            sequences, including nested tuples/frozensets, are invalid. Whole
            Tensors are converted to Python values, with rows becoming tuples.
            Labels must compare equal to themselves (NaN is invalid).
        other_node_labels: Labels aligned with ``other.nodes``. Provide both
            node label arguments, or omit both to disable node label constraints.
            ``None`` entries inside a label sequence remain valid label values.
        edge_labels: Hashable labels aligned with ``graph.edge_index`` rows,
            following the same value rules as ``node_labels``.
        other_edge_labels: Labels aligned with ``other.edge_index`` rows. Provide
            both edge label arguments, or omit both to disable edge label
            constraints. The paired-input rule also applies to empty sequences.
        fixed_mapping: Required source-to-target pairs of actual node IDs.
        max_mappings: Maximum number of structural mappings. Discovering one
            more raises ``ResourceLimitError`` instead of returning a prefix.
        max_states: Maximum number of search states, including the root and
            states rejected by refinement. Both budgets must be positive ints.

    Returns:
        A tensor of shape ``(K, N)`` with target node IDs in source ``nodes``
        order, using the target nodes' dtype and device. No matches yields
        ``(0, N)``; two empty graphs have the unique mapping of shape ``(1, 0)``.
        Traversal is deterministic for the same input; rows are not sorted.

    Joint color refinement and individualization use an explicit search stack
    on CPU. Inputs are copied to CPU once, including CUDA tensors. Edges retain
    their direction, self-loops and multiplicity, including repeated/reversed
    rows of an undirected graph. Graph features are ignored unless explicitly
    passed as labels. There is no chemical interpretation of labels.
    Graphs with different node/edge counts or direction return immediately,
    without inspecting labels or fixed constraints. Reuse graph instances and
    label sequences when comparing the same fragments repeatedly.
    EdgeTable mutation methods keep the graph's node universe synchronized.
    Call ``graph.synchronize_()`` after editing tensor values directly in place.

    Raises:
        ResourceLimitError: A budget is exceeded; no partial result is returned.
        TypeError: Graphs, labels, constraints or budgets have invalid types.
        ValueError: Only one graph's node or edge labels are supplied, labels
            have wrong lengths, constraints reference unknown nodes or repeat a
            target, a budget is not positive, or an edge endpoint is absent from
            an unsynchronized node tensor. Valid but incompatible labels or
            constraints simply produce no matches.

    Examples:
        >>> graph = UndirectedGraph(torch.tensor([[0, 1], [0, 2]]))
        >>> isomorphisms(graph, graph).tolist()
        [[0, 1, 2], [0, 2, 1]]
    """
    _check_limit("max_mappings", max_mappings)
    matches: list[list[int]] = []
    for mapping in _search(
        graph,
        other,
        node_labels,
        other_node_labels,
        edge_labels,
        other_edge_labels,
        fixed_mapping,
        max_states,
    ):
        if len(matches) == max_mappings:
            raise ResourceLimitError("max_mappings", max_mappings)
        matches.append(mapping)
    return other.nodes.new_tensor(matches).reshape(len(matches), graph.nodes.numel())


def automorphisms(
    graph: _Graph,
    *,
    node_labels: _Labels = None,
    edge_labels: _Labels = None,
    fixed_nodes: Sequence[int] | Tensor = (),
    max_mappings: int = 10_000,
    max_states: int = 100_000,
) -> Tensor:
    """Return every automorphism of a graph that fixes the given node IDs.

    Labels, output layout and exact resource budgets follow ``isomorphisms``.
    The identity is included. ``fixed_nodes`` is a sequence of actual node IDs,
    each required to map to itself; repeated fixed nodes are harmless.
    """
    if isinstance(fixed_nodes, Tensor):
        if (
            fixed_nodes.ndim != 1
            or fixed_nodes.dtype == torch.bool
            or fixed_nodes.is_floating_point()
            or fixed_nodes.is_complex()
        ):
            raise TypeError("fixed_nodes must be a 1D integer sequence.")
        fixed_nodes = fixed_nodes.detach().cpu().tolist()
    fixed_mapping = {}
    for node in fixed_nodes:
        _check_node_id(node)
        fixed_mapping[node] = node
    return isomorphisms(
        graph,
        graph,
        node_labels=node_labels,
        other_node_labels=node_labels,
        edge_labels=edge_labels,
        other_edge_labels=edge_labels,
        fixed_mapping=fixed_mapping,
        max_mappings=max_mappings,
        max_states=max_states,
    )


def is_isomorphic(
    graph: _Graph,
    other: _Graph,
    *,
    node_labels: _Labels = None,
    other_node_labels: _Labels = None,
    edge_labels: _Labels = None,
    other_edge_labels: _Labels = None,
    fixed_mapping: Mapping[int, int] | None = None,
    mapping_filter: Callable[[Tensor], bool] | None = None,
    max_states: int = 100_000,
) -> bool:
    """Test whether a label-preserving isomorphism exists, stopping at the first.

    Arguments and mapping layout follow ``isomorphisms``. ``mapping_filter``
    may reject complete structural mappings using domain-specific semantics.
    It receives a target-ID tensor in source node order. Only ``max_states``
    limits this existence search, including candidates rejected by the filter;
    the enumeration interfaces' result limit does not apply. ``False`` means
    the search was exhausted; budget exhaustion raises ``ResourceLimitError``.
    Exceptions raised by the filter propagate to the caller.
    """
    if mapping_filter is not None and not callable(mapping_filter):
        raise TypeError("mapping_filter must be callable.")
    matches = _search(
        graph,
        other,
        node_labels,
        other_node_labels,
        edge_labels,
        other_edge_labels,
        fixed_mapping,
        max_states,
    )
    if mapping_filter is None:
        return next(matches, None) is not None
    output_nodes = None
    for mapping in matches:
        if output_nodes is None:
            output_nodes = other.nodes
        if mapping_filter(output_nodes.new_tensor(mapping)):
            return True
    return False


def _check_node_id(node: int) -> None:
    if isinstance(node, bool) or not isinstance(node, Integral):
        raise TypeError("Fixed node IDs must be integers.")


def _check_limit(name: str, limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, Integral):
        raise TypeError(f"{name} must be an integer.")
    if limit <= 0:
        raise ValueError(f"{name} must be positive.")


def _labels(values: _Labels, size: int) -> list[Hashable]:
    def freeze(value):
        return tuple(freeze(item) for item in value) if isinstance(value, list) else value

    if values is None:
        return [None] * size
    if isinstance(values, Tensor):
        if values.ndim == 0:
            raise ValueError("Labels must have a leading node or edge dimension.")
        values = [freeze(value) for value in values.detach().cpu().tolist()]
    if len(values) != size:
        raise ValueError(f"Expected {size} labels, got {len(values)}.")
    for value in values:
        _check_label(value)
    return list(values)


def _check_label(value: Hashable) -> None:
    if isinstance(value, Tensor):
        raise TypeError("Tensor label items are invalid; pass a whole Tensor or convert labels with .tolist().")
    hash(value)
    if isinstance(value, (tuple, frozenset)):
        for item in value:
            _check_label(item)
    elif value != value:
        raise ValueError("Labels must compare equal to themselves; NaN is invalid.")


def _encode(first: Sequence[Hashable], second: Sequence[Hashable]) -> tuple[list[int], list[int]]:
    palette: dict[Hashable, int] = {}
    encoded = [palette.setdefault(label, len(palette)) for label in (*first, *second)]
    return encoded[: len(first)], encoded[len(first) :]


def _adjacency(graph: _Graph, nodes: list[int], labels: list[int]) -> tuple[_Adjacency, _Adjacency]:
    indices = {node: index for index, node in enumerate(nodes)}
    outgoing: _Adjacency = [[] for _ in nodes]
    directed = isinstance(graph, DirectedGraph)
    incoming: _Adjacency = [[] for _ in nodes] if directed else outgoing
    for (source, target), label in zip(graph.edge_index.detach().cpu().tolist(), labels):
        try:
            source, target = indices[source], indices[target]
        except KeyError as error:
            raise ValueError(
                "Graph edge endpoints are missing from nodes; call graph.synchronize_() after editing edge tensors "
                "in place."
            ) from error
        outgoing[source].append((target, label))
        if directed:
            incoming[target].append((source, label))
        elif source != target:
            outgoing[target].append((source, label))
    return outgoing, incoming


def _refine(
    first: list[int],
    second: list[int],
    adjacency: tuple[tuple[_Adjacency, _Adjacency], ...],
    splitter: int | None = None,
) -> tuple[list[int], list[int]] | None:
    """Split paired color cells by edge counts into active splitter cells.

    A branch starts from a stable parent partition, so only its new singleton
    cell needs activation. Keep the largest part under its old color and queue
    the smaller parts: counts to their complement follow from counts to the
    old cell. This avoids rebuilding every node's signature after every split.
    """
    if Counter(first) != Counter(second):
        return None
    size = len(first)
    colors = first + second
    cells: dict[int, set[int]] = {}
    for node, color in enumerate(colors):
        cells.setdefault(color, set()).add(node)
    pending = set(cells) if splitter is None else {splitter}
    active = deque(sorted(pending))
    next_color = max(cells, default=-1) + 1
    while active:
        color = active.popleft()
        pending.remove(color)
        counts: dict[int, Counter[tuple[int, int]]] = {}
        for node in cells[color]:
            group, local = divmod(node, size)
            outgoing, incoming = adjacency[group]
            # Reverse outgoing arcs to count edges from a node to the splitter;
            # reverse incoming arcs to count edges from the splitter to a node.
            relations = (incoming,) if outgoing is incoming else (incoming, outgoing)
            for direction, edges in enumerate(relations):
                for neighbor, label in edges[local]:
                    target = group * size + neighbor
                    counts.setdefault(target, Counter())[direction, label] += 1
        affected: dict[int, dict[frozenset[tuple[tuple[int, int], int]], set[int]]] = {}
        for node, signature in counts.items():
            by_signature = affected.setdefault(colors[node], {})
            by_signature.setdefault(frozenset(signature.items()), set()).add(node)
        for old_color, by_signature in affected.items():
            parts = list(by_signature.values())
            untouched = cells[old_color].difference(counts)
            if untouched:
                parts.append(untouched)
            if len(parts) == 1:
                continue
            if any(sum(node < size for node in part) * 2 != len(part) for part in parts):
                return None
            # Deterministic tie-breaking uses node positions, never label order.
            largest = max(parts, key=lambda part: (len(part), -min(part)))
            cells[old_color] = largest
            for part in sorted((part for part in parts if part is not largest), key=min):
                cells[next_color] = part
                for node in part:
                    colors[node] = next_color
                active.append(next_color)
                pending.add(next_color)
                next_color += 1
    return colors[:size], colors[size:]


def _branches(
    first: list[int], second: list[int], source: int, targets: list[int]
) -> Iterator[tuple[list[int], list[int], int | None]]:
    color = max(first + second) + 1
    for target in targets:
        left, right = first.copy(), second.copy()
        left[source] = right[target] = color
        yield left, right, color


def _search(
    graph: _Graph,
    other: _Graph,
    node_labels: _Labels,
    other_node_labels: _Labels,
    edge_labels: _Labels,
    other_edge_labels: _Labels,
    fixed_mapping: Mapping[int, int] | None,
    max_states: int,
) -> Iterator[list[int]]:
    _check_limit("max_states", max_states)
    if not isinstance(graph, (DirectedGraph, UndirectedGraph)) or not isinstance(
        other, (DirectedGraph, UndirectedGraph)
    ):
        raise TypeError("graph and other must be DirectedGraph or UndirectedGraph instances.")
    if (
        graph.nodes.numel() != other.nodes.numel()
        or len(graph.edge_index) != len(other.edge_index)
        or isinstance(graph, DirectedGraph) != isinstance(other, DirectedGraph)
    ):
        return
    if (node_labels is None) != (other_node_labels is None):
        raise ValueError("node_labels and other_node_labels must both be provided or both omitted.")
    if (edge_labels is None) != (other_edge_labels is None):
        raise ValueError("edge_labels and other_edge_labels must both be provided or both omitted.")
    nodes = graph.nodes.detach().cpu().tolist()
    other_nodes = nodes if graph is other else other.nodes.detach().cpu().tolist()
    labels = _labels(node_labels, len(nodes))
    other_labels = (
        labels if graph is other and node_labels is other_node_labels else _labels(other_node_labels, len(other_nodes))
    )
    first, second = _encode(labels, other_labels)
    labels = _labels(edge_labels, len(graph.edge_index))
    other_labels = (
        labels
        if graph is other and edge_labels is other_edge_labels
        else _labels(other_edge_labels, len(other.edge_index))
    )
    edge_first, edge_second = _encode(labels, other_labels)
    pins, other_pins = [-1] * len(nodes), [-1] * len(other_nodes)
    if fixed_mapping is not None:
        if not isinstance(fixed_mapping, Mapping):
            raise TypeError("fixed_mapping must be a mapping of source to target node IDs.")
        indices, other_indices = {node: i for i, node in enumerate(nodes)}, {
            node: i for i, node in enumerate(other_nodes)
        }
        for pin, (source, target) in enumerate(fixed_mapping.items()):
            _check_node_id(source)
            _check_node_id(target)
            if source not in indices or target not in other_indices:
                raise ValueError("fixed_mapping references an unknown node ID.")
            if other_pins[other_indices[target]] != -1:
                raise ValueError("fixed_mapping must have distinct target node IDs.")
            pins[indices[source]] = other_pins[other_indices[target]] = pin
    first, second = _encode(list(zip(first, pins)), list(zip(second, other_pins)))
    initial_adjacency = _adjacency(graph, nodes, edge_first)
    adjacency = (
        initial_adjacency,
        (
            initial_adjacency
            if graph is other and edge_labels is other_edge_labels
            else _adjacency(other, other_nodes, edge_second)
        ),
    )
    if len(nodes) <= 1:
        if first == second and Counter(edge_first) == Counter(edge_second):
            yield other_nodes.copy()
        return
    stack: list[Iterator[tuple[list[int], list[int], int | None]]] = [iter(((first, second, None),))]
    states = 0
    while stack:
        state = next(stack[-1], None)
        if state is None:
            stack.pop()
            continue
        states += 1
        if states > max_states:
            raise ResourceLimitError("max_states", max_states)
        refined = _refine(state[0], state[1], adjacency, state[2])
        if refined is None:
            continue
        first, second = refined
        cells: dict[int, list[int]] = {}
        for node, color in enumerate(first):
            cells.setdefault(color, []).append(node)
        ambiguous = [(len(cell), color) for color, cell in cells.items() if len(cell) > 1]
        if ambiguous:
            _, color = min(ambiguous)
            targets = [node for node, label in enumerate(second) if label == color]
            stack.append(_branches(first, second, cells[color][0], targets))
        else:
            targets_by_color = {color: node for node, color in enumerate(second)}
            yield [other_nodes[targets_by_color[color]] for color in first]
