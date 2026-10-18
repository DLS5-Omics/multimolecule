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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


def _adjacency_list(edge_index: torch.Tensor, *, undirected: bool = False) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for u, v in edge_index.tolist():
        adjacency.setdefault(u, set()).add(v)
        if undirected:
            adjacency.setdefault(v, set()).add(u)
    return adjacency


@dataclass
class EdgeTable:
    """Edge index plus aligned feature tensors."""

    edge_index: torch.Tensor
    features: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.edge_index.numel() == 0:
            self.edge_index = self.edge_index.view(0, 2).long()
        if self.edge_index.dim() != 2 or self.edge_index.shape[1] != 2:
            raise ValueError(f"edge index must be (E,2); got shape {tuple(self.edge_index.shape)}")
        if self.features is not None:
            edge_count = self.edge_index.shape[0]
            for key, tensor in self.features.items():
                if tensor.shape[0] != edge_count:
                    raise ValueError(f"edge feature {key} has length {tensor.shape[0]} != edge_count {edge_count}")

    @classmethod
    def empty(cls, device: torch.device | None = None) -> EdgeTable:
        return cls(edge_index=torch.empty((0, 2), dtype=torch.long, device=device), features=None)

    @property
    def num_edges(self) -> int:
        return len(self)

    def to(self, device: torch.device) -> None:
        self.edge_index = self.edge_index.to(device)
        if self.features is not None:
            for key, tensor in self.features.items():
                self.features[key] = tensor.to(device)

    def append(self, edge_index: torch.Tensor, feature: dict[str, Any] | None) -> None:
        """Append a single edge with optional features."""
        self.extend(edge_index.view(1, 2), [feature])

    def extend(self, edge_indices: Iterable[torch.Tensor], features: Iterable[dict[str, Any] | None] | None) -> None:
        """Append a batch, validating features before updating any tensors."""
        if isinstance(edge_indices, torch.Tensor):
            if edge_indices.numel() == 0:
                new_edges = edge_indices.view(0, 2)
            elif edge_indices.dim() == 2 and edge_indices.shape[1] == 2:
                new_edges = edge_indices
            else:
                raise ValueError("edge indices must have shape (E,2)")
            new_edges = new_edges.to(self.edge_index.device, dtype=torch.long)
        else:
            edge_list = list(edge_indices)
            new_edges = (
                torch.cat([edge.view(1, 2).to(self.edge_index.device, dtype=torch.long) for edge in edge_list])
                if edge_list
                else self.edge_index.new_empty((0, 2))
            )
        feat_list = list(features) if features is not None else [None] * len(new_edges)
        if len(new_edges) != len(feat_list):
            raise ValueError("edge_indices and features must have the same length")
        if len(new_edges) == 0:
            return

        feature_chunks: dict[str, list[torch.Tensor]] = {}
        feature_keys = set(self.features) if self.features is not None else None
        for feature in feat_list:
            if feature is None:
                if feature_keys is not None:
                    raise ValueError("feature must be provided for every edge in an EdgeTable with features")
                continue
            if feature_keys is None:
                if feature and (len(self) or any(previous is None for previous in feat_list)):
                    raise ValueError("feature must be provided for every edge in an EdgeTable with features")
                feature_keys = set(feature)
            if set(feature) != feature_keys:
                missing = feature_keys - set(feature)
                extra = set(feature) - feature_keys
                raise ValueError(f"edge feature keys mismatch; missing={missing} extra={extra}")
            for key, value in feature.items():
                tensor = torch.as_tensor(value, device=self.edge_index.device)
                if tensor.dim() == 0 or tensor.shape[0] != 1:
                    tensor = tensor.unsqueeze(0)
                feature_chunks.setdefault(key, []).append(tensor)

        new_features = None
        if feature_keys is not None:
            new_features = {}
            for key, chunks in feature_chunks.items():
                if self.features is not None:
                    existing = self.features[key]
                    chunks = [existing] + [chunk.to(existing.device) for chunk in chunks]
                new_features[key] = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        new_edge_index = torch.cat([self.edge_index, new_edges], dim=0)
        self.edge_index = new_edge_index
        if self.features is None:
            self.features = new_features
        elif new_features is not None:
            self.features.update(new_features)

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
        nodes: set[int] | None = None,
        device: torch.device | None = None,
        edge_features: dict[str, torch.Tensor] | None = None,
        allow_multi_edges: bool = False,
    ):
        self._edges = (
            EdgeTable(edge_index=edge_index, features=edge_features)
            if edge_index is not None
            else EdgeTable.empty(device=device)
        )
        self.nodes = nodes if nodes is not None else set()
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

    def add_node(self, node: int) -> None:
        self.nodes.add(int(node))

    def add_nodes(self, nodes: Iterable[int]) -> None:
        for n in nodes:
            self.add_node(int(n))

    @property
    def num_nodes(self) -> int:
        """Return a best-effort node count (isolated nodes tracked via ``self.nodes``)."""
        if self.nodes:
            return len(self.nodes)
        if self.edge_index.numel() == 0:
            return 0
        return int(torch.max(self.edge_index).item()) + 1

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    def to(self, device: torch.device) -> None:
        """In-place move to a different device."""
        self._edges.to(device)

    def _type_mask(self, edge_type: int | None) -> torch.Tensor | None:
        if edge_type is None or self.edge_features is None or "type" not in self.edge_features:
            return None
        type_tensor = self.edge_features["type"]
        type_values = type_tensor.reshape(-1)
        return type_values == int(edge_type)

    def edge_mask(self, edge_type: int | None = None) -> torch.Tensor:
        """Return a boolean mask over edges, optionally filtered by type."""
        mask = torch.ones(self.edge_index.shape[0], dtype=torch.bool, device=self.edge_index.device)
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return mask

    def edge_indices(self, edge_type: int | None = None) -> list[int]:
        """Return edge indices (row ids) matching the requested type."""
        mask = self.edge_mask(edge_type)
        return torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()

    def has_edge(self, u: int, v: int) -> bool:
        if self.edge_index.numel() == 0:
            return False
        a, b = self._canonical_edge(u, v)
        mask_same = (self.edge_index[:, 0] == a) & (self.edge_index[:, 1] == b)
        return bool(torch.any(mask_same))

    def add_edge(self, u: int, v: int, attr: dict[str, Any] | None = None) -> None:
        u, v = self._canonical_edge(u, v)
        self.add_nodes((u, v))
        if not self.allow_self_loops and u == v:
            return
        if not self.allow_multi_edges and self.has_edge(u, v):
            return
        new_edge = torch.tensor([[u, v]], dtype=torch.long, device=self.edge_index.device)
        self._edges.append(new_edge, attr)

    def add_edges(
        self,
        edges: Iterable[tuple[int, int]],
        attrs: Iterable[dict[str, Any] | None] | None = None,
    ) -> None:
        """Add multiple edges, optionally with attributes; skips duplicates unless multi-edges are allowed."""
        edge_list = [self._canonical_edge(u, v) for u, v in edges]
        attr_list = list(attrs) if attrs is not None else [None] * len(edge_list)
        if len(edge_list) != len(attr_list):
            raise ValueError("edges and attrs must have the same length")
        if not edge_list:
            return
        seen = set(map(tuple, self.edge_index.tolist())) if not self.allow_multi_edges else set()
        accepted_edges = []
        accepted_attrs = []
        for (u, v), attr in zip(edge_list, attr_list):
            if not self.allow_self_loops and u == v:
                continue
            if not self.allow_multi_edges:
                if (u, v) in seen:
                    continue
                seen.add((u, v))
            accepted_edges.append((u, v))
            accepted_attrs.append(attr)
        if accepted_edges:
            edge_index = torch.tensor(accepted_edges, dtype=torch.long, device=self.edge_index.device)
            self._edges.extend(edge_index, accepted_attrs)
        self.add_nodes(node for edge in edge_list for node in edge)

    def _canonical_edge(self, u: int, v: int) -> tuple[int, int]:
        """Canonicalize edge representation; overridden by undirected graphs."""
        return int(u), int(v)

    def reindex_nodes(self, mapping: dict[int, int]) -> None:
        """Relabel nodes simultaneously in one pass over the original edge endpoints."""
        if not mapping:
            return
        mapping = {int(src): int(dst) for src, dst in mapping.items()}
        edge_index = [[mapping.get(u, u), mapping.get(v, v)] for u, v in self.edge_index.tolist()]
        self._edges.edge_index = self.edge_index.new_tensor(edge_index).reshape(-1, 2)
        self.nodes = {mapping.get(node, node) for node in self.nodes}

    def coalesce(self) -> None:
        """Drop duplicate edges (keeping the first occurrence) and align features."""
        if self.edge_index.numel() == 0:
            return
        seen = set()
        keep_mask = []
        for u, v in self.edge_index.tolist():
            key = (u, v)
            if key in seen:
                keep_mask.append(False)
            else:
                seen.add(key)
                keep_mask.append(True)
        keep = torch.tensor(keep_mask, dtype=torch.bool, device=self.edge_index.device)
        self._edges = self._edges.masked_select(keep)

    def edge_subgraph(self, mask: torch.Tensor) -> _BaseGraph:
        """Create a new graph containing only edges where mask is True."""
        new_edges = self._edges.masked_select(mask)
        new_nodes = set(self.nodes)
        if new_edges.edge_index.numel() > 0:
            new_nodes.update(set(torch.unique(new_edges.edge_index).tolist()))
        return self.__class__(
            edge_index=new_edges.edge_index,
            nodes=new_nodes,
            edge_features=new_edges.features,
            allow_multi_edges=self.allow_multi_edges,
        )


class UndirectedGraph(_BaseGraph):
    """Minimal undirected graph helper backed by torch edge indices.

    Nodes are identified by integer ids. Edges are stored as an (E, 2) torch.LongTensor
    with (u, v) where u <= v. A separate node set tracks isolated vertices.
    Neighbor lookups scan the edge table and are O(E).
    """

    allow_self_loops = False

    def _canonical_edge(self, u: int, v: int) -> tuple[int, int]:
        a, b = (int(u), int(v))
        return (a, b) if a <= b else (b, a)

    def remove_node(self, node: int) -> None:
        node = int(node)
        if self.edge_index.numel() > 0:
            mask = (self.edge_index[:, 0] != node) & (self.edge_index[:, 1] != node)
            self._edges.prune_(mask)
        self.nodes.discard(node)

    def neighbors(self, node: int, edge_type: int | None = None) -> list[int]:
        node = int(node)
        if self.edge_index.numel() == 0:
            return []
        mask = (self.edge_index[:, 0] == node) | (self.edge_index[:, 1] == node)
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        incident = self.edge_index[mask]
        neighbors = torch.where(incident[:, 0] == node, incident[:, 1], incident[:, 0])
        return sorted(set(neighbors.tolist()))

    def edge_list(self, edge_type: int | None = None) -> list[tuple[int, int]]:
        if self.edge_index.numel() == 0:
            return []
        mask = torch.ones(self.edge_index.shape[0], dtype=torch.bool, device=self.edge_index.device)
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        filtered = self.edge_index[mask]
        return [(int(u), int(v)) for u, v in filtered.tolist()]

    def degree(self, node: int) -> int:
        return len(self.neighbors(node))

    def degree_by_type(self, node: int) -> dict[int, int]:
        """Count unique neighbors per edge type in one edge-table scan."""
        if self.edge_features is None or "type" not in self.edge_features:
            return {0: self.degree(node)}
        node = int(node)
        neighbors_by_type: dict[int, set[int]] = {}
        types = self.edge_features["type"].reshape(-1).tolist()
        for (u, v), edge_type in zip(self.edge_index.tolist(), types):
            neighbors = neighbors_by_type.setdefault(int(edge_type), set())
            if u == node:
                neighbors.add(v)
            elif v == node:
                neighbors.add(u)
        return {edge_type: len(neighbors) for edge_type, neighbors in sorted(neighbors_by_type.items())}

    def _all_nodes(self) -> list[int]:
        if self.nodes:
            return sorted(self.nodes)
        if self.edge_index.numel() == 0:
            return []
        vals = torch.unique(self.edge_index).tolist()
        return sorted(int(v) for v in vals)

    def connected_components(self) -> list[list[int]]:
        """Return sorted components using an O(V + E) traversal, plus sorting."""
        adjacency = _adjacency_list(self.edge_index, undirected=True)
        comps: list[list[int]] = []
        all_nodes = sorted(self.nodes or adjacency)
        visited: set[int] = set()
        for start in all_nodes:
            if start in visited:
                continue
            visited.add(start)
            stack = [start]
            comp: list[int] = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in adjacency.get(v, ()):
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
            comps.append(sorted(comp))
        return comps

    def subgraph(self, nodes: Sequence[int]) -> UndirectedGraph:
        node_set = {int(n) for n in nodes}
        if self.edge_index.numel() == 0:
            return UndirectedGraph(device=self.edge_index.device, allow_multi_edges=self.allow_multi_edges)
        mask = (torch.isin(self.edge_index[:, 0], torch.tensor(list(node_set), device=self.edge_index.device))) & (
            torch.isin(self.edge_index[:, 1], torch.tensor(list(node_set), device=self.edge_index.device))
        )
        new_edges = self._edges.masked_select(mask)
        return UndirectedGraph(
            edge_index=new_edges.edge_index,
            nodes=set(node_set),
            edge_features=new_edges.features,
            allow_multi_edges=self.allow_multi_edges,
        )

    def is_path_component(self, component: Sequence[int]) -> tuple[bool, list[int]]:
        """Check for a simple path with one edge-table scan and a local traversal."""
        if len(component) <= 1:
            return False, []
        adjacency = _adjacency_list(self.edge_index, undirected=True)
        deg1 = [v for v in component if len(adjacency.get(int(v), ())) == 1]
        if len(deg1) != 2:
            return False, []
        if any(len(adjacency.get(int(v), ())) > 2 for v in component):
            return False, []
        start, end = sorted(deg1)
        path = [start]
        visited = {start}
        current = start
        while current != end:
            nbrs = [n for n in adjacency.get(int(current), ()) if n not in visited]
            if len(nbrs) != 1:
                return False, []
            nxt = nbrs[0]
            path.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(visited) != len(component):
            return False, []
        return True, path


class DirectedGraph(_BaseGraph):
    """Minimal directed graph helper backed by torch edge indices.

    Successor and predecessor lookups scan the edge table and are O(E).
    """

    def successors(self, node: int, edge_type: int | None = None) -> list[int]:
        node = int(node)
        if self.edge_index.numel() == 0:
            return []
        mask = self.edge_index[:, 0] == node
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return sorted(set(self.edge_index[mask, 1].tolist()))

    def predecessors(self, node: int, edge_type: int | None = None) -> list[int]:
        node = int(node)
        if self.edge_index.numel() == 0:
            return []
        mask = self.edge_index[:, 1] == node
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return sorted(set(self.edge_index[mask, 0].tolist()))

    def undirected(self) -> UndirectedGraph:
        g = UndirectedGraph(device=self.edge_index.device, allow_multi_edges=self.allow_multi_edges)
        g.add_nodes(self.nodes)
        if self.edge_index.numel() > 0:
            attrs = None
            if self.edge_features is not None:
                attrs = [{k: v[i] for k, v in self.edge_features.items()} for i in range(self.edge_index.shape[0])]
            g.add_edges(self.edge_index.tolist(), attrs=attrs)
        return g

    def connected_components(self) -> list[list[int]]:
        """Return weakly connected components (via an undirected view)."""
        nodes = self.nodes | set(self.edge_index.reshape(-1).tolist())
        return UndirectedGraph(edge_index=self.edge_index, nodes=nodes).connected_components()

    def is_path_component(self, component: Sequence[int]) -> bool:
        """Check whether the given component induces a simple path in this directed graph."""
        nodes = {int(v) for v in component}
        if len(nodes) <= 1:
            return False
        adjacency = _adjacency_list(self.edge_index)
        path = self._walk_successor_path(min(nodes), adjacency)
        if set(path) != nodes or len(path) != len(nodes):
            return False
        # Last node must also have a successor (reject open-ended paths).
        return bool(adjacency.get(path[-1]))

    def walk_successor_path(self, start: int) -> list[int]:
        """
        Walk a successor-only path starting at ``start`` until a branch, cycle, or dead end.

        Returns the visited node sequence (including ``start``). Stops when out-degree != 1
        or the next node was already visited.
        """
        return self._walk_successor_path(int(start), _adjacency_list(self.edge_index))

    @staticmethod
    def _walk_successor_path(start: int, adjacency: dict[int, set[int]]) -> list[int]:
        path: list[int] = []
        current = start
        visited: set[int] = set()
        while True:
            path.append(current)
            visited.add(current)
            succ = adjacency.get(current, set())
            if len(succ) != 1:
                break
            nxt = next(iter(succ))
            if nxt in visited:
                break
            current = nxt
        return path
