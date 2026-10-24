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


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any
from weakref import WeakSet

import torch
from danling import NestedTensor


def _integer_tensor(values: int | Sequence[int] | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Move IDs to a graph without silently narrowing integer values."""
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, device=reference.device)
        if values.numel() == 0:
            return values.to(dtype=reference.dtype)
    if values.dtype.is_floating_point or values.dtype.is_complex or values.dtype == torch.bool:
        raise TypeError("node and edge IDs must use an integer dtype")
    values = values.to(device=reference.device)
    source, target = torch.iinfo(values.dtype), torch.iinfo(reference.dtype)
    if source.min < target.min or source.max > target.max:
        outside = torch.zeros_like(values, dtype=torch.bool)
        if source.min < target.min:
            outside |= values < target.min
        if source.max > target.max:
            outside |= values > target.max
        if outside.any():
            raise ValueError(f"node and edge IDs are out of range for {reference.dtype}")
    return values.to(dtype=reference.dtype)


def _compact_graph(
    edge_index: torch.Tensor, nodes: Sequence[int] | torch.Tensor = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map arbitrary node IDs to dense indices without moving edges off device."""
    extra = _integer_tensor(nodes, edge_index).reshape(-1)
    node_ids, inverse = torch.unique(torch.cat((edge_index.reshape(-1), extra)), sorted=True, return_inverse=True)
    return node_ids, inverse[: edge_index.numel()].reshape(-1, 2)


def _node_membership(values: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Look up values in sorted node IDs without sorting the queried edge table."""
    if nodes.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    indices = torch.searchsorted(nodes, values.contiguous()).clamp_max(nodes.numel() - 1)
    return nodes[indices] == values


def _merge_nodes(nodes: torch.Tensor, additional: torch.Tensor) -> torch.Tensor:
    """Extend a sorted node universe, sorting only the new candidate IDs first."""
    candidates = torch.unique(additional.reshape(-1), sorted=True)
    added = candidates[~_node_membership(candidates, nodes)]
    if added.numel() == 0:
        return nodes
    return torch.cat((nodes, added)).sort().values


def _component_labels_cpu(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Contract complete CPU parent forests and discard settled quotient edges.

    Every active endpoint names a root before hooking. Parent links only move
    toward smaller roots, and full pointer compression restores that invariant
    before forming the next quotient graph. Quotient self-loops can then be
    dropped permanently. An empty quotient, rather than comparison with an
    aliased old parent tensor, determines when the in-place hooks are done.
    """
    parents = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    jumped = torch.empty_like(parents)
    lower_buffer = parents.new_empty(edge_index.shape[0])
    upper_buffer = torch.empty_like(lower_buffer)
    contracted_buffer = parents.new_empty(edge_index.numel())
    keep_buffer = torch.empty(edge_index.shape[0], dtype=torch.bool, device=edge_index.device)
    active_edges = edge_index
    while active_edges.shape[0]:
        active_count = active_edges.shape[0]
        source, target = active_edges.unbind(1)
        lower, upper = lower_buffer[:active_count], upper_buffer[:active_count]
        torch.minimum(source, target, out=lower)
        torch.maximum(source, target, out=upper)
        parents.scatter_reduce_(0, upper, lower, reduce="amin", include_self=True)
        while True:
            # Distinct buffers preserve all old parent values during a jump.
            torch.index_select(parents, 0, parents, out=jumped)
            if torch.equal(jumped, parents):
                break
            parents, jumped = jumped, parents
        contracted_flat = contracted_buffer[: 2 * active_count]
        torch.index_select(parents, 0, active_edges.reshape(-1), out=contracted_flat)
        contracted = contracted_flat.reshape(-1, 2)
        keep = keep_buffer[:active_count]
        torch.ne(contracted[:, 0], contracted[:, 1], out=keep)
        active_edges = contracted[keep]
    return parents


def _component_labels(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.device.type == "cpu":
        return _component_labels_cpu(edge_index, num_nodes)
    parents = torch.arange(num_nodes, device=edge_index.device)
    source, target = edge_index.unbind(1)
    while True:
        lower = torch.minimum(parents[source], parents[target])
        upper = torch.maximum(parents[source], parents[target])
        updated = parents.scatter_reduce(0, upper, lower, reduce="amin", include_self=True)
        updated = updated[updated]
        if torch.equal(updated, parents):
            return updated
        parents = updated


def _connected_components(
    edge_index: torch.Tensor,
    nodes: Sequence[int] | torch.Tensor = (),
    *,
    node_ids: torch.Tensor | None = None,
) -> NestedTensor:
    # Graph instances already own the complete, sorted, unique node tensor.
    # Reuse it instead of sorting all endpoints again to compact their IDs.
    if node_ids is None:
        node_ids, edges = _compact_graph(edge_index, nodes)
    elif node_ids.numel() == 0:
        node_ids, edges = _compact_graph(edge_index, node_ids)
    elif (node_ids[0] == 0) & (node_ids[-1] == node_ids.numel() - 1):
        edges = edge_index.to(torch.long)
        if ((edges < 0) | (edges >= node_ids.numel())).any():
            node_ids, edges = _compact_graph(edge_index, node_ids)
    else:
        edges = torch.searchsorted(node_ids, edge_index)
        if (node_ids[edges.clamp_max(node_ids.numel() - 1)] != edge_index).any():
            node_ids, edges = _compact_graph(edge_index, node_ids)
    if node_ids.numel() == 0:
        return NestedTensor([], dtype=edge_index.dtype, device=edge_index.device)

    parents = _component_labels(edges, node_ids.numel())

    order = torch.argsort(parents, stable=True)
    _, counts = torch.unique_consecutive(parents[order], return_counts=True)
    # Transfer only lengths once. Keep metadata as tensors and adopt the packed
    # node values directly, without creating one Python object per component.
    counts = counts.to(device="cpu")
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return NestedTensor.from_packed(
        node_ids[order],
        offsets,
        counts.reshape(-1, 1),
        ragged_dims=(0,),
        ragged_offsets=(offsets,),
        validate=False,
        materialize_python_metadata=False,
    )


def _successor_distances(successor: torch.Tensor, start: int | torch.Tensor) -> torch.Tensor:
    """Return first visit times, with ``len(successor)`` for unreachable nodes."""
    size = successor.numel()
    distances = successor.new_full((size,), size)
    if size == 0:
        return distances
    start_index = torch.as_tensor(start, dtype=torch.long, device=successor.device).reshape(1)
    distances.scatter_(0, start_index, 0)
    jump = successor
    # Each round adds the next power-of-two block of visits and doubles the jump.
    for bit in range(size.bit_length()):
        distances.scatter_reduce_(0, jump, distances + (1 << bit), reduce="amin", include_self=True)
        jump = jump[jump]
    return distances


def _directed_path(edge_index: torch.Tensor, start: int | torch.Tensor) -> torch.Tensor:
    start_node = _integer_tensor(start, edge_index).reshape(1)
    nodes, compact_edges = _compact_graph(edge_index, start_node)
    arcs = compact_edges[_first_occurrences(compact_edges)]
    size = nodes.numel()
    degree = torch.bincount(arcs[:, 0], minlength=size)
    successor = arcs.new_full((size,), size)
    successor.scatter_reduce_(0, arcs[:, 0], arcs[:, 1], reduce="amin", include_self=True)
    # Branches and dead ends are absorbing states, so each appears only once.
    successor = torch.where(degree == 1, successor, torch.arange(size, device=edge_index.device))
    distances = _successor_distances(successor, torch.searchsorted(nodes, start_node).reshape(()))
    order = distances.argsort()
    return nodes[order[distances[order] < size]]


def _undirected_path(
    edge_index: torch.Tensor, component: Sequence[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    component = _integer_tensor(component, edge_index).reshape(-1)
    empty = edge_index.new_empty((0,))
    false = edge_index.new_tensor(False, dtype=torch.bool)
    if component.numel() <= 1 or edge_index.numel() == 0:
        return false, empty

    # Unrelated components must not enlarge the arc table or the doubling work.
    # Keep the boundary check: an induced path with an edge leaving the requested
    # component is not a complete path component of the original graph.
    inside = _node_membership(edge_index, component.sort().values)
    closed = ~(inside[:, 0] ^ inside[:, 1]).any()
    edge_index = edge_index[inside.all(dim=1)]
    if edge_index.numel() == 0:
        return false, empty
    nodes, compact_edges = _compact_graph(edge_index, component)
    component_nodes = torch.searchsorted(nodes, component)
    arcs, arc_inverse = torch.unique(torch.cat((compact_edges, compact_edges.flip(1))), dim=0, return_inverse=True)
    size, arc_count, edge_count = nodes.numel(), arcs.shape[0], edge_index.shape[0]
    arc_ids = torch.arange(arc_count, device=edge_index.device)
    degree = torch.bincount(arcs[:, 0], minlength=size)
    component_degree = degree[component_nodes]
    endpoints = component_degree == 1
    start = torch.where(endpoints, component_nodes, size).amin().clamp_max(size - 1)
    end = torch.where(endpoints, component_nodes, -1).amax().clamp_min(0)

    reverse = arc_ids.new_empty((arc_count,))
    reverse.scatter_reduce_(
        0,
        arc_inverse,
        torch.cat((arc_inverse[edge_count:], arc_inverse[:edge_count])),
        reduce="amin",
        include_self=False,
    )
    first_arc = arc_ids.new_full((size,), arc_count)
    first_arc.scatter_reduce_(0, arcs[:, 0], arc_ids, reduce="amin", include_self=True)
    last_arc = arc_ids.new_full((size,), -1)
    last_arc.scatter_reduce_(0, arcs[:, 0], arc_ids, reduce="amax", include_self=True)
    destinations = arcs[:, 1]
    # An oriented edge remembers the previous node, preventing a path from
    # turning back. Degree-two nodes choose the arc other than the reverse arc.
    successor = torch.where(
        degree[destinations] == 2,
        torch.where(first_arc[destinations] == reverse, last_arc[destinations], first_arc[destinations]),
        arc_ids,
    )
    initial_arc = first_arc[start].clamp_max(arc_count - 1)
    distances = _successor_distances(successor, initial_arc)
    reachable = distances < arc_count
    terminal_arc = torch.where(reachable, distances, -1).argmax()
    valid = (
        closed
        & (endpoints.sum() == 2)
        & (component_degree <= 2).all()
        & (start != end)
        & (arcs[terminal_arc, 1] == end)
        & (reachable.sum() + 1 == component.numel())
    )
    order = distances.argsort()
    order = order[reachable[order]]
    path = nodes[torch.cat((start.reshape(1), arcs[order, 1]))]
    valid &= torch.isin(path, component).all()
    return valid, path.masked_select(valid)


def _first_occurrences(rows: torch.Tensor) -> torch.Tensor:
    """Select each (E, 2) edge row's first occurrence, preserving edge order."""
    if rows.shape[0] <= 1:
        return torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
    # Stable lexicographic sorting avoids generic row comparisons
    # inside torch.unique(dim=0), and retains the original first duplicate.
    order = rows[:, 1].argsort(stable=True)
    order = order[rows[order, 0].argsort(stable=True)]
    ordered = rows[order]
    first = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
    first[1:] = (ordered[1:] != ordered[:-1]).any(dim=1)
    keep = torch.zeros_like(first)
    keep[order[first]] = True
    return keep


@dataclass
class _AdjacencyIndex:
    """Sorted sparse rows, retaining original edge IDs for feature filtering."""

    sources: torch.Tensor
    offsets: torch.Tensor
    targets: torch.Tensor
    edge_ids: torch.Tensor
    counts: torch.Tensor
    typed_counts: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def bounds(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sources.numel() == 0:
            empty = torch.zeros_like(nodes, dtype=torch.long)
            return empty, empty
        positions = torch.searchsorted(self.sources, nodes.contiguous()).clamp_max(self.sources.numel() - 1)
        present = self.sources[positions] == nodes
        return torch.where(present, self.offsets[positions], 0), torch.where(present, self.offsets[positions + 1], 0)


def _build_adjacency(edge_index: torch.Tensor, *, reverse: bool = False, undirected: bool = False) -> _AdjacencyIndex:
    source, target = edge_index.unbind(1)
    edge_ids = torch.arange(edge_index.shape[0], dtype=torch.long, device=edge_index.device)
    if undirected:
        # A loop contributes one incident edge and one neighbor.
        non_loop = source != target
        source, target = torch.cat((source, target[non_loop])), torch.cat((target, source[non_loop]))
        edge_ids = torch.cat((edge_ids, edge_ids[non_loop]))
    elif reverse:
        source, target = target, source
    order = target.argsort(stable=True)
    order = order[source[order].argsort(stable=True)]
    source, target, edge_ids = source[order], target[order], edge_ids[order]
    sources, inverse, lengths = torch.unique_consecutive(source, return_inverse=True, return_counts=True)
    first = torch.ones(source.numel(), dtype=torch.bool, device=edge_index.device)
    first[1:] = (source[1:] != source[:-1]) | (target[1:] != target[:-1])
    counts = torch.bincount(inverse[first], minlength=sources.numel())
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    return _AdjacencyIndex(sources, offsets, target, edge_ids, counts)


def _lookup_counts(query: torch.Tensor, sources: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if sources.numel() == 0:
        return torch.zeros_like(query, dtype=torch.long)
    positions = torch.searchsorted(sources, query.contiguous()).clamp_max(sources.numel() - 1)
    return torch.where(sources[positions] == query, counts[positions], 0)


@dataclass
class _SuccessorIndex:
    successor: torch.Tensor
    parents: torch.Tensor
    offsets: torch.Tensor
    order: torch.Tensor


@dataclass
class EdgeTable:
    """Edge index plus aligned feature tensors.

    Graphs track successful calls to :meth:`append`, :meth:`extend`,
    :meth:`prune_` and :meth:`to`, and replacement of ``edge_index``. Direct
    edits to tensor values or feature dictionaries require an explicit call to
    the owning graph's ``synchronize_()`` before further graph operations.
    Standalone tables preserve the supplied edge orientation and multiplicity.
    """

    edge_index: torch.Tensor
    features: dict[str, torch.Tensor] | None = None
    _revision: int = field(default=0, init=False, repr=False, compare=False)
    _owners: WeakSet[_BaseGraph] = field(default_factory=WeakSet, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "_owners"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._owners = WeakSet()

    def _notify_owners(self, origin: _BaseGraph | None = None) -> None:
        for owner in tuple(self._owners):
            if owner is not origin:
                owner._synchronize_edges()

    def __post_init__(self) -> None:
        if self.edge_index.numel() == 0:
            self.edge_index = self.edge_index.view(0, 2)
        if self.edge_index.dim() != 2 or self.edge_index.shape[1] != 2:
            raise ValueError(f"edge index must be (E,2); got shape {tuple(self.edge_index.shape)}")
        if (
            self.edge_index.dtype.is_floating_point
            or self.edge_index.dtype.is_complex
            or self.edge_index.dtype == torch.bool
        ):
            raise TypeError("edge index must use an integer dtype")
        self.features = self.features or None
        if self.features is not None:
            edge_count = self.edge_index.shape[0]
            for key, tensor in self.features.items():
                if tensor.ndim == 0 or tensor.shape[0] != edge_count:
                    raise ValueError(f"edge feature {key} must have leading dimension {edge_count}")
                if tensor.device != self.edge_index.device:
                    raise ValueError(f"edge feature {key} must be on {self.edge_index.device}")

    @classmethod
    def empty(cls, device: torch.device | None = None) -> EdgeTable:
        return cls(edge_index=torch.empty((0, 2), dtype=torch.long, device=device), features=None)

    @property
    def num_edges(self) -> torch.Tensor:
        return self.edge_index.new_tensor(len(self), dtype=torch.long)

    def to(self, device: torch.device, *, _origin: _BaseGraph | None = None) -> None:
        edge_index = self.edge_index.to(device)
        features = (
            {key: tensor.to(device) for key, tensor in self.features.items()} if self.features is not None else {}
        )
        self.edge_index = edge_index
        if self.features is not None:
            self.features.update(features)
        self._revision += 1
        self._notify_owners(_origin)

    def append(self, edge_index: torch.Tensor, feature: dict[str, Any] | None) -> None:
        """Append a single edge with optional features."""
        features = (
            {key: torch.as_tensor(value, device=self.edge_index.device).unsqueeze(0) for key, value in feature.items()}
            if feature is not None
            else None
        )
        self.extend(edge_index.reshape(1, 2), features)

    def extend(
        self,
        edge_indices: torch.Tensor,
        features: dict[str, torch.Tensor] | None = None,
        *,
        _origin: _BaseGraph | None = None,
    ) -> None:
        """Append a batch, validating features before updating any tensors."""
        if edge_indices.ndim != 2 or edge_indices.shape[1] != 2:
            raise ValueError("edge indices must have shape (E,2)")
        if edge_indices.dtype.is_floating_point or edge_indices.dtype.is_complex or edge_indices.dtype == torch.bool:
            raise TypeError("edge indices must use an integer dtype")
        new_edges = _integer_tensor(edge_indices, self.edge_index)
        features = features or None
        normalized = None
        if features is not None:
            normalized = {}
            for key, tensor in features.items():
                if tensor.ndim == 0 or tensor.shape[0] != new_edges.shape[0]:
                    raise ValueError(f"edge feature {key} must have leading dimension {new_edges.shape[0]}")
                normalized[key] = tensor.to(device=self.edge_index.device)
        if len(new_edges) == 0:
            return
        if self.features is not None:
            if normalized is None:
                raise ValueError("feature must be provided for every edge in an EdgeTable with features")
            if normalized.keys() != self.features.keys():
                raise ValueError("edge feature keys mismatch")
            normalized = {key: torch.cat((existing, normalized[key])) for key, existing in self.features.items()}
        elif normalized and len(self):
            raise ValueError("feature must be provided for every edge in an EdgeTable with features")

        new_edge_index = torch.cat((self.edge_index, new_edges))
        self.edge_index = new_edge_index
        self.features = normalized
        self._revision += 1
        self._notify_owners(_origin)

    def prune_(self, mask: torch.Tensor, *, _origin: _BaseGraph | None = None) -> None:
        """In-place keep only edges where mask is True."""
        edge_index = self.edge_index[mask]
        features = {key: tensor[mask] for key, tensor in self.features.items()} if self.features is not None else {}
        self.edge_index = edge_index
        if self.features is not None:
            self.features.update(features)
        self._revision += 1
        self._notify_owners(_origin)

    def masked_select(self, mask: torch.Tensor) -> EdgeTable:
        new_features = None
        if self.features is not None:
            new_features = {k: v[mask] for k, v in self.features.items()}
        return EdgeTable(edge_index=self.edge_index[mask], features=new_features)

    def __len__(self) -> int:
        return len(self.edge_index)


class _BaseGraph:
    """Shared scaffolding for simple graph types backed by torch edge indices.

    Successful EdgeTable mutations synchronize graph bookkeeping immediately;
    replacing its edge_index tensor is detected before the next graph read.
    Registered nodes remain present when edges are pruned. Unchanged reads
    only compare mutation metadata, without scanning the edge table. After
    editing tensor values or feature dictionaries directly, call
    :meth:`synchronize_` explicitly.
    """

    allow_self_loops: bool = True
    allow_multi_edges: bool = False
    _nodes: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor | None = None,
        nodes: torch.Tensor | None = None,
        device: torch.device | None = None,
        edge_features: dict[str, torch.Tensor] | None = None,
        allow_multi_edges: bool = False,
        num_nodes: int | None = None,
    ):
        if edge_index is None:
            edge_index = torch.empty((0, 2), dtype=torch.long, device=device)
        self._edges = EdgeTable(edge_index=self._canonical_edges(edge_index), features=edge_features)
        if num_nodes is None:
            explicit_nodes = (
                _integer_tensor(nodes, self.edge_index).reshape(-1)
                if nodes is not None
                else self.edge_index.new_empty((0,))
            )
            self.nodes = torch.unique(torch.cat((explicit_nodes, self.edge_index.reshape(-1))), sorted=True)
        else:
            if nodes is not None:
                raise ValueError("nodes and num_nodes cannot both be specified")
            if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
                raise TypeError("num_nodes must be an integer")
            if num_nodes < 0:
                raise ValueError("num_nodes must be non-negative")
            maximum = torch.iinfo(self.edge_index.dtype).max
            if num_nodes > maximum + 1:
                raise ValueError(f"node IDs are out of range for {self.edge_index.dtype}")
            outside = self.edge_index < 0
            if num_nodes <= maximum:
                outside |= self.edge_index >= num_nodes
            if outside.any():
                raise ValueError(f"edge endpoints must be in [0, {num_nodes})")
            self.nodes = torch.arange(num_nodes, dtype=self.edge_index.dtype, device=self.edge_index.device)
        self.allow_multi_edges = allow_multi_edges
        self._acknowledge_edges()

    def _acknowledge_edges(self) -> None:
        """Record graph bookkeeping already applied to the current edge table."""
        previous = getattr(self, "_synchronized_table", None)
        if previous is not None and previous is not self._edges:
            previous._owners.discard(self)
        self._edges._owners.add(self)
        self._synchronized_table = self._edges
        self._synchronized_tensor = self._edges.edge_index
        self._synchronized_revision = self._edges._revision
        self._invalidate_indices()

    def _invalidate_indices(self) -> None:
        self._adjacency: dict[bool, _AdjacencyIndex] = {}
        self._successor_index: _SuccessorIndex | None = None
        self._edge_type_groups: tuple[torch.Tensor, torch.Tensor] | None = None

    def _adjacency_index(self, reverse: bool = False) -> _AdjacencyIndex:
        self._synchronize_edges()
        if reverse not in self._adjacency:
            self._adjacency[reverse] = _build_adjacency(
                self._edges.edge_index, reverse=reverse, undirected=isinstance(self, UndirectedGraph)
            )
        return self._adjacency[reverse]

    def _neighbor_rows(self, node: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        index = self._adjacency_index(reverse)
        begin, end = index.bounds(node.reshape(()))
        return index.targets[begin:end], index.edge_ids[begin:end]

    def _neighbors(self, node: torch.Tensor, edge_type: int | None, reverse: bool = False) -> torch.Tensor:
        edges = self._edges.edge_index
        if edges.device.type == "cuda":
            # Reading CSR row bounds as slice endpoints synchronizes twice.
            # A device-side scan is faster for the measured single-node CUDA
            # queries; batch degrees and walks still reuse prepared indices.
            undirected = isinstance(self, UndirectedGraph)
            mask = (edges[:, 0] == node) | (edges[:, 1] == node) if undirected else edges[:, int(reverse)] == node
            type_mask = self._type_mask(edge_type)
            if type_mask is not None:
                mask &= type_mask
            if undirected:
                incident = edges[mask]
                targets = torch.where(incident[:, 0] == node, incident[:, 1], incident[:, 0])
            else:
                targets = edges[mask, 1 - int(reverse)]
            return torch.unique(targets, sorted=True)
        targets, edge_ids = self._neighbor_rows(node, reverse)
        features = self._edges.features
        if edge_type is not None and features is not None and "type" in features:
            if isinstance(edge_type, torch.Tensor):
                edge_type = edge_type.to(device=targets.device)
            targets = targets[features["type"].reshape(-1)[edge_ids] == edge_type]
        return torch.unique_consecutive(targets)

    def _incident_edge_ids(self, nodes: torch.Tensor) -> torch.Tensor:
        """Gather only requested sparse rows, retaining edges crossing their boundary."""
        index = self._adjacency_index()
        begins, ends = index.bounds(nodes.reshape(-1))
        lengths = ends - begins
        group_ends = lengths.cumsum(0)
        positions = torch.repeat_interleave(begins - (group_ends - lengths), lengths)
        positions += torch.arange(positions.numel(), device=self.device)
        return torch.unique(index.edge_ids[positions], sorted=True)

    def _synchronize_edges(self, force: bool = False) -> None:
        # Constructors can access edge_index before the initial node universe
        # exists. Read raw attributes here to avoid recursive graph properties.
        if not hasattr(self, "_nodes"):
            return
        table = self._edges
        if (
            not force
            and getattr(self, "_synchronized_table", None) is table
            and getattr(self, "_synchronized_tensor", None) is table.edge_index
            and getattr(self, "_synchronized_revision", -1) == table._revision
        ):
            return
        table.__post_init__()
        edge_index = table.edge_index
        # Retain already canonical storage so synchronizing one of several
        # owners does not repeatedly invalidate every other owner's snapshot.
        if isinstance(self, UndirectedGraph) and (edge_index[:, 0] > edge_index[:, 1]).any():
            edge_index = self._canonical_edges(edge_index)
        nodes = _integer_tensor(self._nodes, edge_index).reshape(-1)
        if force:
            nodes = torch.unique(nodes, sorted=True)
        nodes = _merge_nodes(nodes, edge_index)
        table.edge_index = edge_index
        self._nodes = nodes
        self._acknowledge_edges()

    def __getstate__(self) -> dict[str, Any]:
        self._synchronize_edges()
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_synchronized_")
            and key not in ("_adjacency", "_successor_index", "_edge_type_groups")
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = state.copy()
        if "nodes" in state:
            state.setdefault("_nodes", state.pop("nodes"))
        self.__dict__.update(state)
        self._invalidate_indices()
        self._synchronize_edges()

    def synchronize_(self) -> None:
        """Restore graph invariants after directly editing mutable tensors.

        EdgeTable mutation methods and replacing its ``edge_index`` tensor are
        detected automatically. Editing tensor values in place, including the
        ``nodes`` tensor, or editing feature dictionaries is not tracked; call
        this method afterwards. Existing node IDs are retained as isolated
        nodes when they no longer occur in the edge table. Undirected edges
        are canonicalized without removing loops, duplicate rows or features.
        Other graphs sharing the edge table are also notified.
        """
        self._synchronize_edges(force=True)
        self._edges._revision += 1
        self._acknowledge_edges()
        self._edges._notify_owners(self)

    @property
    def nodes(self) -> torch.Tensor:
        self._synchronize_edges()
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: torch.Tensor) -> None:
        self._nodes = nodes
        self._invalidate_indices()

    @property
    def edge_index(self) -> torch.Tensor:
        self._synchronize_edges()
        return self._edges.edge_index

    @property
    def edge_features(self) -> dict[str, torch.Tensor] | None:
        self._synchronize_edges()
        return self._edges.features

    @property
    def edges(self) -> EdgeTable:
        self._synchronize_edges()
        return self._edges

    def add_node(self, node: int | torch.Tensor) -> None:
        self.add_nodes(_integer_tensor(node, self.edge_index).reshape(1))

    def add_nodes(self, nodes: torch.Tensor) -> None:
        self.nodes = _merge_nodes(self.nodes, _integer_tensor(nodes, self.nodes))

    @property
    def num_nodes(self) -> torch.Tensor:
        """Return the number of distinct nodes as a scalar integer tensor."""
        return self.nodes.new_tensor(self.nodes.numel(), dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    def to(self, device: torch.device) -> None:
        """In-place move to a different device."""
        self._synchronize_edges()
        self._edges.to(device, _origin=self)
        self._nodes = self._nodes.to(device)
        self._acknowledge_edges()

    def _type_mask(self, edge_type: int | None) -> torch.Tensor | None:
        if edge_type is None or self.edge_features is None or "type" not in self.edge_features:
            return None
        type_tensor = self.edge_features["type"]
        type_values = type_tensor.reshape(-1)
        return type_values == torch.as_tensor(edge_type, device=self.device)

    def edge_mask(self, edge_type: int | None = None) -> torch.Tensor:
        """Return a boolean mask over edges, optionally filtered by type."""
        mask = torch.ones(self.edge_index.shape[0], dtype=torch.bool, device=self.edge_index.device)
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return mask

    def edge_indices(self, edge_type: int | None = None) -> torch.Tensor:
        """Return edge indices (row ids) matching the requested type."""
        mask = self._type_mask(edge_type)
        if mask is None:
            return torch.arange(self.edge_index.shape[0], dtype=torch.long, device=self.device)
        return torch.nonzero(mask, as_tuple=False).reshape(-1)

    def edge_list(self, edge_type: int | None = None) -> torch.Tensor:
        """Return selected edge rows as an (E,2) tensor on the graph device."""
        mask = self._type_mask(edge_type)
        return self.edge_index.clone() if mask is None else self.edge_index[mask]

    def has_edge(self, u: int | torch.Tensor, v: int | torch.Tensor) -> torch.Tensor:
        source = _integer_tensor(u, self.edge_index)
        target = _integer_tensor(v, self.edge_index).reshape(())
        if self._edges.edge_index.device.type == "cuda":
            query = torch.stack((source.reshape(()), target)).reshape(1, 2)
            return (self._edges.edge_index == self._canonical_edges(query)).all(dim=1).any()
        targets, _ = self._neighbor_rows(source)
        if targets.numel() == 0:
            return target.new_tensor(False, dtype=torch.bool)
        position = torch.searchsorted(targets, target).clamp_max(targets.numel() - 1)
        return targets[position] == target

    def add_edge(self, u: int | torch.Tensor, v: int | torch.Tensor, attr: dict[str, Any] | None = None) -> None:
        edge = torch.stack((_integer_tensor(u, self.edge_index), _integer_tensor(v, self.edge_index)))
        attrs = (
            {key: torch.as_tensor(value, device=self.device).unsqueeze(0) for key, value in attr.items()}
            if attr is not None
            else None
        )
        self.add_edges(edge.reshape(1, 2), attrs)

    def add_edges(
        self,
        edges: torch.Tensor,
        attrs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Add multiple edges, optionally with attributes; skips duplicates unless multi-edges are allowed."""
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (E,2)")
        if edges.dtype.is_floating_point or edges.dtype.is_complex or edges.dtype == torch.bool:
            raise TypeError("edges must use an integer dtype")
        edges = self._canonical_edges(_integer_tensor(edges, self.edge_index))
        if attrs is not None:
            for key, tensor in attrs.items():
                if tensor.ndim == 0 or tensor.shape[0] != edges.shape[0]:
                    raise ValueError(f"edge feature {key} must have leading dimension {edges.shape[0]}")
        keep = torch.ones(edges.shape[0], dtype=torch.bool, device=self.device)
        if not self.allow_self_loops:
            keep &= edges[:, 0] != edges[:, 1]
        if not self.allow_multi_edges:
            if edges.shape[0] == 1:
                keep &= ~(self.edge_index == edges[0]).all(dim=1).any()
            else:
                keep &= _first_occurrences(torch.cat((self.edge_index, edges)))[len(self._edges) :]
        new_nodes = _merge_nodes(self.nodes, edges)
        accepted_attrs = (
            {key: tensor.to(device=self.device)[keep] for key, tensor in attrs.items()} if attrs is not None else None
        )
        self._edges.extend(edges[keep], accepted_attrs, _origin=self)
        self.nodes = new_nodes
        self._acknowledge_edges()

    def _canonical_edges(self, edges: torch.Tensor) -> torch.Tensor:
        """Canonicalize edge representation; overridden by undirected graphs."""
        return edges

    def reindex_nodes(self, mapping: torch.Tensor) -> None:
        """Relabel nodes simultaneously in one pass over the original edge endpoints."""
        if mapping.ndim != 2 or mapping.shape[1] != 2:
            raise ValueError("mapping must have shape (K,2)")
        if mapping.shape[0] == 0:
            return
        if mapping.dtype.is_floating_point or mapping.dtype.is_complex or mapping.dtype == torch.bool:
            raise TypeError("mapping must use an integer dtype")
        mapping = _integer_tensor(mapping, self.edge_index)
        mapping = mapping[mapping[:, 0].argsort()]
        if torch.unique(mapping[:, 0]).numel() != mapping.shape[0]:
            raise ValueError("mapping source node IDs must be unique")

        def relabel(values: torch.Tensor) -> torch.Tensor:
            indices = torch.searchsorted(mapping[:, 0].contiguous(), values.reshape(-1)).clamp_max(mapping.shape[0] - 1)
            return torch.where(
                mapping[indices, 0] == values.reshape(-1), mapping[indices, 1], values.reshape(-1)
            ).reshape(values.shape)

        edge_index = self._canonical_edges(relabel(self.edge_index))
        nodes = torch.unique(relabel(self.nodes), sorted=True)
        self._edges.edge_index = edge_index
        self._edges._revision += 1
        self.nodes = nodes
        self._acknowledge_edges()
        self._edges._notify_owners(self)

    def coalesce(self) -> None:
        """Drop duplicate edges (keeping the first occurrence) and align features."""
        self._edges = self._edges.masked_select(_first_occurrences(self.edge_index))
        self._acknowledge_edges()

    def edge_subgraph(self, mask: torch.Tensor) -> _BaseGraph:
        """Create a new graph containing only edges where mask is True."""
        new_edges = self._edges.masked_select(mask)
        graph_type = UndirectedGraph if isinstance(self, UndirectedGraph) else DirectedGraph
        return graph_type(
            edge_index=new_edges.edge_index,
            nodes=self.nodes.clone(),
            edge_features=new_edges.features,
            allow_multi_edges=self.allow_multi_edges,
        )


class UndirectedGraph(_BaseGraph):
    """Minimal undirected graph helper backed by torch edge indices.

    Nodes are identified by integer ids. Edges are stored as an (E, 2) torch.LongTensor
    with (u, v) where u <= v. A sorted node tensor tracks isolated vertices.
    Pass ``num_nodes=N`` for the node universe ``0..N-1`` to skip ID sorting;
    otherwise ``nodes`` can supply arbitrary IDs and isolated vertices.
    CPU neighbor queries reuse an adjacency index built on demand. Mutations
    clear the index; subsequent queries rebuild it from the current edge table.
    Single-node CUDA neighbor queries use vectorized edge scans.
    """

    allow_self_loops = False

    def _canonical_edges(self, edges: torch.Tensor) -> torch.Tensor:
        return edges.sort(dim=1).values

    def remove_node(self, node: int | torch.Tensor) -> None:
        node = _integer_tensor(node, self.edge_index)
        nodes = self.nodes[self.nodes != node]
        if self.edge_index.numel() > 0:
            mask = (self.edge_index[:, 0] != node) & (self.edge_index[:, 1] != node)
            self._edges.prune_(mask, _origin=self)
        self.nodes = nodes
        self._acknowledge_edges()

    def neighbors(self, node: int | torch.Tensor, edge_type: int | None = None) -> torch.Tensor:
        node = _integer_tensor(node, self.edge_index)
        return self._neighbors(node, edge_type)

    def degree(self, node: int | torch.Tensor) -> torch.Tensor:
        query = _integer_tensor(node, self.edge_index).reshape(())
        index = self._adjacency_index()
        return _lookup_counts(query, index.sources, index.counts)

    def degrees(
        self, nodes: int | Sequence[int] | torch.Tensor | None = None, edge_type: int | None = None
    ) -> torch.Tensor:
        """Count unique neighbors for many nodes using a reusable adjacency index.

        By default, return counts aligned with ``self.nodes``. Explicit query IDs
        retain their shape and order, including repeats and nodes absent from the
        graph. Duplicate edges and self-loops follow :meth:`neighbors` semantics.
        """
        query = self.nodes if nodes is None else _integer_tensor(nodes, self.nodes)
        index = self._adjacency_index()
        features = self._edges.features
        if edge_type is None or features is None or "type" not in features:
            return _lookup_counts(query, index.sources, index.counts)
        # Scalar Tensor queries are mutable and hash by object identity. They
        # retain comparison support, but must not become persistent cache keys.
        cacheable = not isinstance(edge_type, torch.Tensor)
        if not cacheable or edge_type not in index.typed_counts:
            if isinstance(edge_type, torch.Tensor):
                edge_type = edge_type.to(device=self.device)
            mask = features["type"].reshape(-1)[index.edge_ids] == edge_type
            # Each source row is already sorted by target, including duplicates.
            sources = torch.repeat_interleave(index.sources, index.offsets.diff())[mask]
            targets = index.targets[mask]
            first = torch.ones(sources.numel(), dtype=torch.bool, device=self.device)
            first[1:] = (sources[1:] != sources[:-1]) | (targets[1:] != targets[:-1])
            selected, counts = torch.unique_consecutive(sources[first], return_counts=True)
            if selected.numel() == 0:
                return torch.zeros_like(query, dtype=torch.long)
            if not cacheable:
                return _lookup_counts(query, selected, counts)
            index.typed_counts[edge_type] = selected, counts
        return _lookup_counts(query, *index.typed_counts[edge_type])

    def degree_by_type(self, node: int | torch.Tensor) -> torch.Tensor:
        """Return sorted (type, unique-neighbor count) rows, including zero counts."""
        if self.edge_features is None or "type" not in self.edge_features:
            degree = self.degree(node)
            return torch.stack((torch.zeros_like(degree), degree)).reshape(1, 2)
        node = _integer_tensor(node, self.edge_index)
        if self._edge_type_groups is None:
            self._edge_type_groups = torch.unique(
                self.edge_features["type"].reshape(-1), sorted=True, return_inverse=True
            )
        types, type_indices = self._edge_type_groups
        neighbors, edge_ids = self._neighbor_rows(node)
        unique = torch.unique(torch.stack((type_indices[edge_ids], neighbors), dim=1), dim=0)
        counts = torch.bincount(unique[:, 0], minlength=types.numel())
        return torch.stack((types.to(torch.long), counts), dim=1)

    def connected_components(self) -> NestedTensor:
        """Return sorted component tensors on the graph device, packed in a NestedTensor.

        Parent hooking and pointer jumping operate on sparse edge tensors, with
        a scalar convergence check each round. Component sizes are copied to CPU
        once for tensor-backed ragged metadata; packed node data stay on device.
        Explicit isolated nodes and all edge endpoints are included.
        """
        return _connected_components(self.edge_index, node_ids=self.nodes)

    def subgraph(self, nodes: torch.Tensor) -> UndirectedGraph:
        nodes = torch.unique(_integer_tensor(nodes, self.nodes).reshape(-1), sorted=True)
        mask = torch.isin(self.edge_index, nodes).all(dim=1)
        new_edges = self._edges.masked_select(mask)
        return UndirectedGraph(
            edge_index=new_edges.edge_index,
            nodes=nodes,
            edge_features=new_edges.features,
            allow_multi_edges=self.allow_multi_edges,
        )

    def is_path_component(self, component: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a scalar bool tensor and ordered path tensor on the graph device.

        The path starts at the smaller endpoint and is empty when the component
        is not a simple path. Degrees count unique neighbors in the full graph.
        """
        component = _integer_tensor(component, self.edge_index).reshape(-1)
        relevant = self.edge_index[self._incident_edge_ids(component)]
        return _undirected_path(relevant, component)


class DirectedGraph(_BaseGraph):
    """Minimal directed graph helper backed by torch edge indices.

    CPU successor and predecessor lookups reuse sparse adjacency indices, built
    on demand and invalidated by graph mutations. Single-node CUDA lookups use
    vectorized edge scans.
    ``num_nodes=N`` declares the node universe ``0..N-1`` without ID sorting;
    ``nodes`` instead supplies arbitrary IDs and isolated vertices.
    """

    def successors(self, node: int | torch.Tensor, edge_type: int | None = None) -> torch.Tensor:
        node = _integer_tensor(node, self.edge_index)
        return self._neighbors(node, edge_type)

    def predecessors(self, node: int | torch.Tensor, edge_type: int | None = None) -> torch.Tensor:
        node = _integer_tensor(node, self.edge_index)
        return self._neighbors(node, edge_type, reverse=True)

    def undirected(self) -> UndirectedGraph:
        edges = self.edge_index.sort(dim=1).values
        keep = edges[:, 0] != edges[:, 1]
        if not self.allow_multi_edges:
            keep &= _first_occurrences(edges)
        features = (
            {key: value[keep] for key, value in self.edge_features.items()} if self.edge_features is not None else None
        )
        return UndirectedGraph(
            edge_index=edges[keep],
            nodes=self.nodes.clone(),
            edge_features=features,
            allow_multi_edges=self.allow_multi_edges,
        )

    def connected_components(self) -> NestedTensor:
        """Return sorted weak component tensors on the graph device as a NestedTensor."""
        return _connected_components(self.edge_index, node_ids=self.nodes)

    def is_path_component(self, component: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return a scalar bool tensor for the successor path check on the graph device."""
        nodes = torch.unique(_integer_tensor(component, self.edge_index))
        false = self.edge_index.new_tensor(False, dtype=torch.bool)
        if nodes.numel() <= 1:
            return false
        # Only sources in the requested component can affect this predicate.
        # Retain outside destinations so a unique successor leaving it remains
        # visible, and preserve terminal branches accepted by the path contract.
        relevant = self.edge_index[self._incident_edge_ids(nodes)]
        path = _directed_path(relevant, nodes[0])
        if path.numel() != nodes.numel():
            return false
        # Last node must also have a successor (reject open-ended paths).
        return (path.sort().values == nodes).all() & (relevant[:, 0] == path[-1]).any()

    def walk_successor_path(self, start: int | torch.Tensor) -> torch.Tensor:
        """
        Walk a successor-only path starting at ``start`` until a branch, cycle, or dead end.

        Returns a node tensor on the graph device, including ``start``. Stops
        when out-degree != 1 or the next node was already visited. Sparse
        successor doubling avoids a Python loop and device sync per node.
        Functional component membership is prepared once and reused, so later
        walks only double the component containing ``start``.
        """
        start_node = _integer_tensor(start, self.edge_index).reshape(())
        adjacent = self._adjacency_index()
        if self._nodes.numel() == 0:
            return start_node.reshape(1).clone()
        start_index = torch.searchsorted(self._nodes, start_node).clamp_max(self._nodes.numel() - 1)
        if self._nodes[start_index] != start_node:
            return start_node.reshape(1).clone()
        if _lookup_counts(start_node, adjacent.sources, adjacent.counts) != 1:
            return start_node.reshape(1).clone()
        if self._successor_index is None:
            size = self._nodes.numel()
            node_indices = torch.arange(size, dtype=torch.long, device=self.device)
            successor = node_indices.clone()
            single = adjacent.counts == 1
            sources = torch.searchsorted(self._nodes, adjacent.sources[single])
            targets = adjacent.targets[adjacent.offsets[:-1][single]]
            successor[sources] = torch.searchsorted(self._nodes, targets)
            edges = torch.stack((node_indices, successor), dim=1)
            parents = _component_labels(edges[node_indices != successor], size)
            order = parents.argsort(stable=True)
            components, lengths = torch.unique_consecutive(parents[order], return_counts=True)
            offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            # Remap root IDs to contiguous component row IDs once.
            component_ids = torch.searchsorted(components, parents)
            self._successor_index = _SuccessorIndex(successor, component_ids, offsets, order)
        index = self._successor_index
        component_id = index.parents[start_index]
        component = index.order[index.offsets[component_id] : index.offsets[component_id + 1]]
        successor = torch.searchsorted(component, index.successor[component])
        distances = _successor_distances(successor, torch.searchsorted(component, start_index))
        order = distances.argsort()
        return self._nodes[component[order[distances[order] < component.numel()]]]
