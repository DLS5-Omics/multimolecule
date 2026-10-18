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
from dataclasses import dataclass
from numbers import Integral
from typing import Any

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

    if edge_index.device.type == "cpu":
        parents = _component_labels_cpu(edges, node_ids.numel())
    else:
        parents = torch.arange(node_ids.numel(), device=edge_index.device)
        source, target = edges.unbind(1)
        while True:
            # Hook the larger parent to the smaller parent, then shorten parent chains.
            lower = torch.minimum(parents[source], parents[target])
            upper = torch.maximum(parents[source], parents[target])
            updated = parents.scatter_reduce(0, upper, lower, reduce="amin", include_self=True)
            updated = updated[updated]
            if torch.equal(updated, parents):
                break
            parents = updated

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
    arcs = torch.unique(compact_edges, dim=0)
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
        (endpoints.sum() == 2)
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
    """Select the first occurrence of each row while preserving row order."""
    _, inverse = torch.unique(rows, dim=0, return_inverse=True)
    positions = torch.arange(rows.shape[0], device=rows.device)
    first = torch.full_like(positions, rows.shape[0])
    first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    return positions == first[inverse]


@dataclass
class EdgeTable:
    """Edge index plus aligned feature tensors."""

    edge_index: torch.Tensor
    features: dict[str, torch.Tensor] | None = None

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

    def to(self, device: torch.device) -> None:
        self.edge_index = self.edge_index.to(device)
        if self.features is not None:
            for key, tensor in self.features.items():
                self.features[key] = tensor.to(device)

    def append(self, edge_index: torch.Tensor, feature: dict[str, Any] | None) -> None:
        """Append a single edge with optional features."""
        features = (
            {key: torch.as_tensor(value, device=self.edge_index.device).unsqueeze(0) for key, value in feature.items()}
            if feature is not None
            else None
        )
        self.extend(edge_index.reshape(1, 2), features)

    def extend(self, edge_indices: torch.Tensor, features: dict[str, torch.Tensor] | None = None) -> None:
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

    def prune_(self, mask: torch.Tensor) -> None:
        """In-place keep only edges where mask is True."""
        self.edge_index = self.edge_index[mask]
        if self.features is None:
            return
        for key, tensor in self.features.items():
            self.features[key] = tensor[mask]

    def masked_select(self, mask: torch.Tensor) -> EdgeTable:
        new_features = None
        if self.features is not None:
            new_features = {k: v[mask] for k, v in self.features.items()}
        return EdgeTable(edge_index=self.edge_index[mask], features=new_features)

    def __len__(self) -> int:
        return len(self.edge_index)


class _BaseGraph:
    """Shared scaffolding for simple graph types backed by torch edge indices."""

    allow_self_loops: bool = True
    allow_multi_edges: bool = False

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

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edges.edge_index

    @property
    def edge_features(self) -> dict[str, torch.Tensor] | None:
        return self._edges.features

    @property
    def edges(self) -> EdgeTable:
        return self._edges

    def add_node(self, node: int | torch.Tensor) -> None:
        self.add_nodes(_integer_tensor(node, self.edge_index).reshape(1))

    def add_nodes(self, nodes: torch.Tensor) -> None:
        self.nodes = torch.unique(torch.cat((self.nodes, _integer_tensor(nodes, self.nodes).reshape(-1))), sorted=True)

    @property
    def num_nodes(self) -> torch.Tensor:
        """Return the number of distinct nodes as a scalar integer tensor."""
        return self.nodes.new_tensor(self.nodes.numel(), dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    def to(self, device: torch.device) -> None:
        """In-place move to a different device."""
        self._edges.to(device)
        self.nodes = self.nodes.to(device)

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
        mask = self.edge_mask(edge_type)
        return torch.nonzero(mask, as_tuple=False).reshape(-1)

    def edge_list(self, edge_type: int | None = None) -> torch.Tensor:
        """Return selected edge rows as an (E,2) tensor on the graph device."""
        return self.edge_index[self.edge_mask(edge_type)]

    def has_edge(self, u: int | torch.Tensor, v: int | torch.Tensor) -> torch.Tensor:
        query = torch.stack((_integer_tensor(u, self.edge_index), _integer_tensor(v, self.edge_index))).reshape(1, 2)
        return (self.edge_index == self._canonical_edges(query)).all(dim=1).any()

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
            keep &= _first_occurrences(torch.cat((self.edge_index, edges)))[len(self._edges) :]
        new_nodes = torch.unique(torch.cat((self.nodes, edges.reshape(-1))), sorted=True)
        accepted_attrs = (
            {key: tensor.to(device=self.device)[keep] for key, tensor in attrs.items()} if attrs is not None else None
        )
        self._edges.extend(edges[keep], accepted_attrs)
        self.nodes = new_nodes

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
        self.nodes = nodes

    def coalesce(self) -> None:
        """Drop duplicate edges (keeping the first occurrence) and align features."""
        self._edges = self._edges.masked_select(_first_occurrences(self.edge_index))

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
    Neighbor lookups scan the edge table and are O(E).
    """

    allow_self_loops = False

    def _canonical_edges(self, edges: torch.Tensor) -> torch.Tensor:
        return edges.sort(dim=1).values

    def remove_node(self, node: int | torch.Tensor) -> None:
        node = _integer_tensor(node, self.edge_index)
        if self.edge_index.numel() > 0:
            mask = (self.edge_index[:, 0] != node) & (self.edge_index[:, 1] != node)
            self._edges.prune_(mask)
        self.nodes = self.nodes[self.nodes != node]

    def neighbors(self, node: int | torch.Tensor, edge_type: int | None = None) -> torch.Tensor:
        node = _integer_tensor(node, self.edge_index)
        mask = (self.edge_index[:, 0] == node) | (self.edge_index[:, 1] == node)
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        incident = self.edge_index[mask]
        neighbors = torch.where(incident[:, 0] == node, incident[:, 1], incident[:, 0])
        return torch.unique(neighbors, sorted=True)

    def degree(self, node: int | torch.Tensor) -> torch.Tensor:
        return self.edge_index.new_tensor(self.neighbors(node).numel(), dtype=torch.long)

    def degree_by_type(self, node: int | torch.Tensor) -> torch.Tensor:
        """Return sorted (type, unique-neighbor count) rows, including zero counts."""
        if self.edge_features is None or "type" not in self.edge_features:
            degree = self.degree(node)
            return torch.stack((torch.zeros_like(degree), degree)).reshape(1, 2)
        node = _integer_tensor(node, self.edge_index)
        types, type_indices = torch.unique(self.edge_features["type"].reshape(-1), sorted=True, return_inverse=True)
        source, target = self.edge_index.unbind(1)
        incident = (source == node) | (target == node)
        neighbors = torch.where(source[incident] == node, target[incident], source[incident])
        unique = torch.unique(torch.stack((type_indices[incident], neighbors), dim=1), dim=0)
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
        return _undirected_path(self.edge_index, component)


class DirectedGraph(_BaseGraph):
    """Minimal directed graph helper backed by torch edge indices.

    Successor and predecessor lookups scan the edge table and are O(E).
    ``num_nodes=N`` declares the node universe ``0..N-1`` without ID sorting;
    ``nodes`` instead supplies arbitrary IDs and isolated vertices.
    """

    def successors(self, node: int | torch.Tensor, edge_type: int | None = None) -> torch.Tensor:
        node = _integer_tensor(node, self.edge_index)
        mask = self.edge_index[:, 0] == node
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return torch.unique(self.edge_index[mask, 1], sorted=True)

    def predecessors(self, node: int | torch.Tensor, edge_type: int | None = None) -> torch.Tensor:
        node = _integer_tensor(node, self.edge_index)
        mask = self.edge_index[:, 1] == node
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return torch.unique(self.edge_index[mask, 0], sorted=True)

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
        path = self.walk_successor_path(nodes[0])
        if path.numel() != nodes.numel():
            return false
        # Last node must also have a successor (reject open-ended paths).
        return (path.sort().values == nodes).all() & (self.edge_index[:, 0] == path[-1]).any()

    def walk_successor_path(self, start: int | torch.Tensor) -> torch.Tensor:
        """
        Walk a successor-only path starting at ``start`` until a branch, cycle, or dead end.

        Returns a node tensor on the graph device, including ``start``. Stops
        when out-degree != 1 or the next node was already visited. Sparse
        successor doubling avoids a Python loop and device sync per node.
        """
        return _directed_path(self.edge_index, start)
