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
        edge_index = edge_index.view(1, 2).to(self.edge_index.device, dtype=torch.long)
        self.edge_index = torch.cat([self.edge_index, edge_index], dim=0)
        if feature is None:
            if self.features is None:
                return
            raise ValueError("feature must be provided when appending to an EdgeTable with existing features")
        normalized: dict[str, torch.Tensor] = {}
        for k, v in feature.items():
            t = torch.as_tensor(v, device=self.edge_index.device)
            if t.dim() == 0 or t.shape[0] != 1:
                t = t.unsqueeze(0)
            normalized[k] = t
        if self.features is None:
            self.features = dict(normalized)
            return
        if set(normalized) != set(self.features):
            missing = set(self.features) - set(normalized)
            extra = set(normalized) - set(self.features)
            raise ValueError(f"edge feature keys mismatch; missing={missing} extra={extra}")
        for key, tensor in normalized.items():
            self.features[key] = torch.cat([self.features[key], tensor.to(self.features[key].device)], dim=0)

    def extend(self, edge_indices: Iterable[torch.Tensor], features: Iterable[dict[str, Any] | None] | None) -> None:
        if features is None:
            for edge_idx in edge_indices:
                self.append(edge_idx, None)
            return
        edge_list = list(edge_indices)
        feat_list = list(features)
        if len(edge_list) != len(feat_list):
            raise ValueError("edge_indices and features must have the same length")
        for edge_idx, feat in zip(edge_list, feat_list):
            self.append(edge_idx, feat)

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
        if attrs is None:
            for u, v in edges:
                self.add_edge(u, v, attr=None)
            return
        edge_list = list(edges)
        attr_list = list(attrs)
        if len(edge_list) != len(attr_list):
            raise ValueError("edges and attrs must have the same length")
        for (u, v), attr in zip(edge_list, attr_list):
            self.add_edge(u, v, attr=attr)

    def _canonical_edge(self, u: int, v: int) -> tuple[int, int]:
        """Canonicalize edge representation; overridden by undirected graphs."""
        return int(u), int(v)

    def reindex_nodes(self, mapping: dict[int, int]) -> None:
        """Relabel nodes and update edges in-place according to a mapping."""
        if not mapping:
            return
        idx = self.edge_index.clone()
        for src, dst in mapping.items():
            idx[idx == int(src)] = int(dst)
        self._edges.edge_index = idx
        self.nodes = {mapping.get(n, n) for n in self.nodes}

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
        neigh = set()
        if self.edge_index.numel() > 0:
            mask0 = self.edge_index[:, 0] == node
            mask1 = self.edge_index[:, 1] == node
            mask = mask0 | mask1
            type_mask = self._type_mask(edge_type)
            if type_mask is not None:
                mask = mask & type_mask
            if torch.any(mask0):
                neigh.update(self.edge_index[mask & mask0, 1].tolist())
            if torch.any(mask1):
                neigh.update(self.edge_index[mask & mask1, 0].tolist())
        return sorted(neigh)

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
        """Return degree counts split by edge type (when type feature exists)."""
        counts: dict[int, int] = {}
        if self.edge_features is None or "type" not in self.edge_features:
            counts[0] = self.degree(node)
            return counts
        types = torch.unique(self.edge_features["type"].reshape(-1)).tolist()
        for edge_type in types:
            counts[int(edge_type)] = len(self.neighbors(node, edge_type=edge_type))
        return counts

    def _all_nodes(self) -> list[int]:
        if self.nodes:
            return sorted(self.nodes)
        if self.edge_index.numel() == 0:
            return []
        vals = torch.unique(self.edge_index).tolist()
        return sorted(int(v) for v in vals)

    def connected_components(self) -> list[list[int]]:
        """Return connected components with deterministic (sorted) ordering."""
        comps: list[list[int]] = []
        all_nodes = sorted(self._all_nodes())
        visited: set[int] = set()
        for start in all_nodes:
            if start in visited:
                continue
            stack = [start]
            comp: list[int] = []
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                comp.append(v)
                for w in self.neighbors(v):
                    if w not in visited:
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
        """Check whether the given component induces a simple path in this undirected graph."""
        if len(component) == 1:
            return False, []
        deg1 = [v for v in component if self.degree(v) == 1]
        if len(deg1) != 2:
            return False, []
        if any(self.degree(v) > 2 for v in component):
            return False, []
        start, end = sorted(deg1)
        path = [start]
        visited = {start}
        current = start
        while current != end:
            nbrs = [n for n in self.neighbors(current) if n not in visited]
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
        return sorted(set(self.edge_index[mask, 1].tolist())) if torch.any(mask) else []

    def predecessors(self, node: int, edge_type: int | None = None) -> list[int]:
        node = int(node)
        if self.edge_index.numel() == 0:
            return []
        mask = self.edge_index[:, 1] == node
        type_mask = self._type_mask(edge_type)
        if type_mask is not None:
            mask = mask & type_mask
        return sorted(set(self.edge_index[mask, 0].tolist())) if torch.any(mask) else []

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
        return self.undirected().connected_components()

    def is_path_component(self, component: Sequence[int]) -> bool:
        """Check whether the given component induces a simple path in this directed graph."""
        nodes = {int(v) for v in component}
        if len(nodes) <= 1:
            return False
        start = sorted(nodes)[0]
        path = self.walk_successor_path(start)
        if set(path) != nodes or len(path) != len(nodes):
            return False
        # Last node must also have a successor (reject open-ended paths).
        return len(self.successors(path[-1])) > 0

    def walk_successor_path(self, start: int) -> list[int]:
        """
        Walk a successor-only path starting at ``start`` until a branch, cycle, or dead end.

        Returns the visited node sequence (including ``start``). Stops when out-degree != 1
        or the next node was already visited.
        """
        path: list[int] = []
        current = int(start)
        visited: set[int] = set()
        while True:
            path.append(current)
            visited.add(current)
            succ = self.successors(current)
            if len(succ) != 1:
                break
            nxt = succ[0]
            if nxt in visited:
                break
            current = nxt
        return path
