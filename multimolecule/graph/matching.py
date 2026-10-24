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

import sys
from numbers import Integral

import torch
from torch import Tensor


def maximum_weight_matching(edge_index: Tensor, edge_weight: Tensor, num_nodes: int | None = None) -> Tensor:
    """Compute an exact maximum-weight matching with tensor Edmonds blossom state.

    ``edge_index`` has shape ``(E, 2)`` and integer endpoints; ``edge_weight``
    has shape ``(E,)``. Non-positive/non-finite weights and self-loops are
    ignored, and repeated undirected edges keep their maximum weight. The
    returned canonical edges retain the endpoint tensor's dtype and device.

    Graph data, blossom membership, labels, duals, and matching stay on the
    input device. Edge scans and primal-dual updates are tensor operations;
    blossom-tree traversal uses scalar synchronization for control flow.
    Independent connected components are solved separately. Sparse adjacency,
    edge state, and blossom trees use O(V + E) auxiliary space.
    The objective maximizes weight, without a maximum-cardinality tie-break.
    """
    if not isinstance(edge_index, Tensor) or not isinstance(edge_weight, Tensor):
        raise TypeError("edge_index and edge_weight must be torch tensors.")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"edge_index must have shape (E, 2), but got {edge_index.shape}.")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[0]:
        raise ValueError("edge_weight must be a 1D tensor aligned with edge_index.")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if edge_index.dtype not in integer_dtypes:
        raise TypeError("edge_index must contain integer endpoints.")
    if edge_weight.dtype not in integer_dtypes and not edge_weight.is_floating_point():
        raise TypeError("edge_weight must contain real numeric values.")
    if edge_weight.device != edge_index.device:
        raise ValueError("edge_index and edge_weight must be on the same device.")
    if num_nodes is not None:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
            raise TypeError("num_nodes must be an integer.")
        if num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, but got {num_nodes}.")
    if (edge_index < 0).any():
        raise ValueError("edge_index must contain non-negative endpoints.")
    non_loop = edge_index[:, 0] != edge_index[:, 1]
    if (
        num_nodes is not None
        and num_nodes <= torch.iinfo(edge_index.dtype).max
        and (edge_index[non_loop] >= num_nodes).any()
    ):
        raise ValueError(f"edge is out of bounds for num_nodes={num_nodes}.")
    keep = non_loop & torch.isfinite(edge_weight) & (edge_weight > 0)
    edges = edge_index[keep].sort(dim=1).values
    weights = edge_weight[keep].detach()
    if edges.shape[0] == 0:
        return edge_index.new_empty((0, 2))
    edges, inverse = torch.unique(edges, dim=0, sorted=True, return_inverse=True)
    integer_weights = not edge_weight.is_floating_point()
    weights = weights.to(torch.int64 if integer_weights else torch.float64)
    reduced = weights.new_zeros(edges.shape[0])
    reduced.scatter_reduce_(0, inverse, weights, reduce="amax", include_self=True)
    nodes, compact = torch.unique(edges.reshape(-1), sorted=True, return_inverse=True)
    compact = compact.reshape(-1, 2)
    if (torch.bincount(compact.reshape(-1), minlength=nodes.numel()) <= 1).all():
        return edges
    # Import lazily: graph.__init__ exports matching before topology.
    from .topology import _connected_components

    components = _connected_components(compact, ())
    node_order, shapes = components.concatenate()
    node_order = node_order.reshape(-1)
    sizes = compact.new_tensor([shape[0] for shape in shapes])
    component_ids = torch.arange(len(shapes), device=nodes.device)
    component_by_node = compact.new_empty(nodes.numel())
    component_by_node[node_order] = torch.repeat_interleave(component_ids, sizes, output_size=nodes.numel())
    edge_component = component_by_node[compact[:, 0]]
    large_components = [index for index, shape in enumerate(shapes) if shape[0] > 3]
    selected = []
    # A connected component with at most three vertices admits only one
    # matching edge. Solve all such components together, including ties.
    if len(large_components) < len(shapes):
        small = sizes <= 3
        maximum = reduced.new_zeros(len(shapes))
        maximum.scatter_reduce_(0, edge_component, reduced, reduce="amax", include_self=True)
        if integer_weights:
            if (maximum[sizes == 3] > torch.iinfo(torch.int64).max // 4).any():
                raise ValueError("integer edge weights exceed the exact int64 primal-dual arithmetic range.")
        else:
            wide = (sizes == 3) & (maximum > torch.finfo(reduced.dtype).max / 4)
            if wide.any() and (((reduced / 4) * 4 != reduced) & wide[edge_component]).any():
                raise ValueError("edge weight dynamic range exceeds exact floating primal-dual arithmetic.")
        eligible = small[edge_component] & (reduced == maximum[edge_component])
        positions = torch.arange(edges.shape[0], device=nodes.device)
        first = positions.new_full((len(shapes),), edges.shape[0])
        first.scatter_reduce_(0, edge_component[eligible], positions[eligible], reduce="amin", include_self=True)
        selected.append(compact[first[small]])
    if large_components:
        edge_order = edge_component.argsort(stable=True)
        counts = torch.bincount(edge_component, minlength=len(shapes))
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    for index in large_components:
        component = components[index]
        # Scalar offsets control device slices; edge/node values stay on device.
        group = edge_order[int(offsets[index]) : int(offsets[index + 1])]
        local_edges = torch.searchsorted(component, compact[group])
        local_weights = reduced[group]
        # Connected simple graphs with V-1 edges are trees. Their matching
        # recurrence needs only local gains, with no blossom or dual events.
        if nodes.device.type == "cpu" and local_edges.shape[0] == component.numel() - 1:
            mate = _tree_matching_cpu(local_edges, local_weights, component.numel())
        else:
            mate = _TensorBlossom(local_edges, local_weights, component.numel()).solve()
        vertices = torch.arange(component.numel(), device=nodes.device)
        first = vertices[(mate >= 0) & (vertices < mate)]
        selected.append(component[torch.stack((first, mate[first]), dim=1)])
    matched = torch.cat(selected)
    return nodes[matched[matched[:, 0].argsort()]]


def _tree_matching_cpu(edges: Tensor, weights: Tensor, num_nodes: int) -> Tensor:
    """Solve a tree in linear space using exact local matching gains."""
    maximum = weights.max()
    if not weights.is_floating_point():
        if maximum > torch.iinfo(torch.int64).max // 4:
            raise ValueError("integer edge weights exceed the exact int64 primal-dual arithmetic range.")
    elif maximum > torch.finfo(weights.dtype).max / 4:
        scaled = weights / 4
        if not torch.equal(scaled * 4, weights):
            raise ValueError("edge weight dynamic range exceeds exact floating primal-dual arithmetic.")
        weights = scaled
    source, target = edges.unbind(1)
    sources, neighbors = torch.cat((source, target)), torch.cat((target, source))
    adjacent_edges = torch.arange(edges.shape[0], device=edges.device).repeat(2)
    permutation = sources.argsort(stable=True)
    neighbors, adjacent_edges = neighbors[permutation], adjacent_edges[permutation]
    counts = torch.bincount(sources, minlength=num_nodes)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    parent = edges.new_full((num_nodes,), -1)
    parent_edge = torch.empty_like(parent)
    order = torch.empty_like(parent)
    order[0] = 0
    end = 1
    for position in range(num_nodes):
        vertex = int(order[position])
        start, stop = int(offsets[vertex]), int(offsets[vertex + 1])
        adjacent = neighbors[start:stop]
        forward = adjacent != parent[vertex]
        children = adjacent[forward]
        parent[children] = vertex
        parent_edge[children] = adjacent_edges[start:stop][forward]
        order[end : end + children.numel()] = children
        end += children.numel()
    # If F(v) is the best subtree matching and B(v) forbids matching v to a
    # child, F(v)-B(v) = max(0, max_child(weight(v,c)-(F(c)-B(c)))).
    # Store this difference instead of large sums of independent subtrees;
    # int64 accuracy is retained even when the total objective would overflow.
    gain = weights.new_zeros(num_nodes)
    choice = edges.new_full((num_nodes,), -1)
    for position in range(num_nodes - 1, 0, -1):
        vertex = int(order[position])
        ancestor = int(parent[vertex])
        candidate = weights[parent_edge[vertex]] - gain[vertex]
        if candidate > gain[ancestor]:
            gain[ancestor] = candidate
            choice[ancestor] = vertex
    mate = edges.new_full((num_nodes,), -1)
    for position in range(num_nodes):
        vertex = int(order[position])
        child = int(choice[vertex])
        if child >= 0 and mate[vertex] < 0:
            mate[vertex], mate[child] = child, vertex
    return mate


class _TensorBlossom:
    """Indexed weighted Edmonds search with persistent graph state in tensors."""

    def __init__(self, edges: Tensor, weights: Tensor, num_nodes: int):
        self.edges = edges
        self.n = num_nodes
        self.capacity = 2 * self.n
        self.device = edges.device
        self.integer_weights = not weights.is_floating_point()
        maximum = weights.max()
        if self.integer_weights:
            if maximum > torch.iinfo(torch.int64).max // 4:
                raise ValueError("integer edge weights exceed the exact int64 primal-dual arithmetic range.")
        elif maximum > torch.finfo(weights.dtype).max / 4:
            # A power-of-two rescaling preserves every representable weight
            # exactly unless the component spans an extreme subnormal range.
            scaled = weights / 4
            if not torch.equal(scaled * 4, weights):
                raise ValueError("edge weight dynamic range exceeds exact floating primal-dual arithmetic.")
            weights = scaled
        self.weights = weights
        self.infinity = weights.new_tensor(torch.iinfo(torch.int64).max if self.integer_weights else torch.inf)
        n, capacity = self.n, self.capacity
        self.vertices = torch.arange(n, device=self.device)
        self.ids = torch.arange(capacity, device=self.device)
        self.mate = edges.new_full((n,), -1)
        self.label = edges.new_zeros(capacity)
        self.label_edge = edges.new_full((capacity, 2), -1)
        self.in_blossom = self.vertices.clone()
        self.parent = edges.new_full((capacity,), -1)
        self.base = self.ids.clone()
        self.active = self.ids < n
        self.children = [edges.new_empty(0)] * capacity
        self.cycle_edges = [edges.new_empty((0, 2))] * capacity
        self.dual = weights.new_zeros(capacity)
        self.dual[:n] = weights.max()
        self.allowed = torch.zeros(edges.shape[0], dtype=torch.bool, device=self.device)
        self.queued = torch.zeros(n, dtype=torch.bool, device=self.device)
        source, target = edges.unbind(1)
        sources, neighbors = torch.cat((source, target)), torch.cat((target, source))
        # Preserve the previous ascending-neighbor search order without a
        # potentially overflowing source * n + target sorting key.
        order = neighbors.argsort(stable=True)
        order = order[sources[order].argsort(stable=True)]
        self.neighbors = neighbors[order]
        self.adjacent_edges = torch.arange(edges.shape[0], device=self.device).repeat(2)[order]
        counts = torch.bincount(sources, minlength=n)
        self.offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    def leaves(self, blossom: int) -> Tensor:
        return _blossom_leaves(blossom, self.n, self.vertices, self.children)

    def allow_edge(self, source: int, target: int) -> None:
        start, end = int(self.offsets[source]), int(self.offsets[source + 1])
        self.allowed[self.adjacent_edges[start:end][self.neighbors[start:end] == target]] = True

    def assign_label(self, node: int, value: int, through: int = -1) -> None:
        blossom = int(self.in_blossom[node])
        self.label[node] = self.label[blossom] = value
        if through < 0:
            self.label_edge[node] = self.label_edge[blossom] = -1
        else:
            self.label_edge[node, 0] = self.label_edge[blossom, 0] = through
            self.label_edge[node, 1] = self.label_edge[blossom, 1] = node
        if value == 1:
            self.queued[self.leaves(blossom)] = True
        else:
            base = int(self.base[blossom])
            self.assign_label(int(self.mate[base]), 1, base)

    def scan_blossom(self, v: int, w: int) -> int:
        trace = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        base = -1
        while v >= 0:
            blossom = int(self.in_blossom[v])
            if self.label[blossom] & 4:
                base = int(self.base[blossom])
                break
            trace[blossom] = True
            self.label[blossom] = 5
            through = int(self.label_edge[blossom, 0])
            if through < 0:
                v = -1
            else:
                blossom = int(self.in_blossom[through])
                v = int(self.label_edge[blossom, 0])
            if w >= 0:
                v, w = w, v
        self.label[trace] = 1
        return base

    def add_blossom(self, base: int, v: int, w: int) -> None:
        base_blossom = int(self.in_blossom[base])
        v_blossom, w_blossom = int(self.in_blossom[v]), int(self.in_blossom[w])
        blossom = int(torch.nonzero(~self.active & (self.ids >= self.n), as_tuple=False)[0, 0])
        self.active[blossom] = True
        self.base[blossom] = base
        self.parent[blossom] = -1
        self.parent[base_blossom] = blossom
        children = self.edges.new_empty(self.capacity)
        cycle_edges = self.edges.new_empty((self.capacity, 2))
        count, edge_count = 0, 1
        cycle_edges[0, 0] = v
        cycle_edges[0, 1] = w
        while v_blossom != base_blossom:
            self.parent[v_blossom] = blossom
            children[count] = v_blossom
            count += 1
            cycle_edges[edge_count] = self.label_edge[v_blossom]
            edge_count += 1
            v = int(self.label_edge[v_blossom, 0])
            v_blossom = int(self.in_blossom[v])
        children[count] = base_blossom
        count += 1
        children[:count] = children[:count].flip(0)
        cycle_edges[:edge_count] = cycle_edges[:edge_count].flip(0)
        while w_blossom != base_blossom:
            self.parent[w_blossom] = blossom
            children[count] = w_blossom
            count += 1
            cycle_edges[edge_count] = self.label_edge[w_blossom].flip(0)
            edge_count += 1
            w = int(self.label_edge[w_blossom, 0])
            w_blossom = int(self.in_blossom[w])
        self.children[blossom] = children[:count].clone()
        self.cycle_edges[blossom] = cycle_edges[:edge_count].clone()
        self.label[blossom] = 1
        self.label_edge[blossom] = self.label_edge[base_blossom]
        self.dual[blossom] = 0
        leaves = self.leaves(blossom)
        self.queued[leaves] |= self.label[self.in_blossom[leaves]] == 2
        self.in_blossom[leaves] = blossom

    def expand_blossom(self, blossom: int, end_stage: bool) -> None:
        def recurse(current: int, end: bool):
            count = self.children[current].numel()
            for child_tensor in self.children[current]:
                child = int(child_tensor)
                self.parent[child] = -1
                if child >= self.n:
                    if end and self.dual[child] == 0:
                        yield child
                    else:
                        self.in_blossom[self.leaves(child)] = child
                else:
                    self.in_blossom[child] = child
            if not end and self.label[current] == 2:
                entry_child = int(self.in_blossom[self.label_edge[current, 1]])
                idx = int(torch.nonzero(self.children[current] == entry_child, as_tuple=False)[0, 0])
                step = 1 if idx & 1 else -1
                if idx & 1:
                    idx -= count
                v, w = int(self.label_edge[current, 0]), int(self.label_edge[current, 1])
                while idx != 0:
                    edge = self.cycle_edges[current][idx % count if step == 1 else (idx - 1) % count]
                    p, q = (int(edge[0]), int(edge[1])) if step == 1 else (int(edge[1]), int(edge[0]))
                    self.label[w] = self.label[q] = 0
                    self.assign_label(w, 2, v)
                    self.allow_edge(p, q)
                    idx += step
                    edge = self.cycle_edges[current][idx % count if step == 1 else (idx - 1) % count]
                    v, w = (int(edge[0]), int(edge[1])) if step == 1 else (int(edge[1]), int(edge[0]))
                    self.allow_edge(v, w)
                    idx += step
                child = int(self.children[current][idx % count])
                self.label[w] = self.label[child] = 2
                self.label_edge[w, 0] = self.label_edge[child, 0] = v
                self.label_edge[w, 1] = self.label_edge[child, 1] = w
                idx += step
                while int(self.children[current][idx % count]) != entry_child:
                    child = int(self.children[current][idx % count])
                    if self.label[child] != 1:
                        members = self.leaves(child)
                        leaves = members[self.label[members] != 0]
                        if leaves.numel():
                            x = int(leaves[0])
                            self.label[x] = self.label[self.mate[self.base[child]]] = 0
                            self.assign_label(x, 2, int(self.label_edge[x, 0]))
                    idx += step
            self.active[current] = False
            self.label[current] = 0
            self.label_edge[current] = -1
            self.parent[current] = -1
            self.base[current] = -1
            self.dual[current] = 0
            self.children[current] = self.edges.new_empty(0)
            self.cycle_edges[current] = self.edges.new_empty((0, 2))

        stack = [recurse(blossom, end_stage)]
        while stack:
            for child in stack[-1]:
                stack.append(recurse(child, end_stage))
                break
            else:
                stack.pop()

    def augment_blossom(self, blossom: int, node: int) -> None:
        def recurse(current: int, x: int):
            child = x
            while int(self.parent[child]) != current:
                child = int(self.parent[child])
            if child >= self.n:
                yield child, x
            count = self.children[current].numel()
            start = int(torch.nonzero(self.children[current] == child, as_tuple=False)[0, 0])
            idx, step = start, 1 if start & 1 else -1
            if start & 1:
                idx -= count
            while idx != 0:
                idx += step
                child = int(self.children[current][idx % count])
                edge = self.cycle_edges[current][idx % count if step == 1 else (idx - 1) % count]
                w, y = (int(edge[0]), int(edge[1])) if step == 1 else (int(edge[1]), int(edge[0]))
                if child >= self.n:
                    yield child, w
                idx += step
                child = int(self.children[current][idx % count])
                if child >= self.n:
                    yield child, y
                self.mate[w] = y
                self.mate[y] = w
            self.children[current] = self.children[current].roll(-start)
            self.cycle_edges[current] = self.cycle_edges[current].roll(-start, dims=0)
            self.base[current] = self.base[self.children[current][0]]

        stack = [recurse(blossom, node)]
        while stack:
            for child, x in stack[-1]:
                stack.append(recurse(child, x))
                break
            else:
                stack.pop()

    def augment_matching(self, v: int, w: int) -> None:
        for node, target in ((v, w), (w, v)):
            while True:
                blossom = int(self.in_blossom[node])
                if blossom >= self.n:
                    self.augment_blossom(blossom, node)
                self.mate[node] = target
                matched = int(self.label_edge[blossom, 0])
                if matched < 0:
                    break
                other = int(self.in_blossom[matched])
                node, target = int(self.label_edge[other, 0]), int(self.label_edge[other, 1])
                if other >= self.n:
                    self.augment_blossom(other, target)
                self.mate[target] = node

    def update_duals(self) -> int:
        kind, index = _blossom_update_duals(
            self.edges,
            self.weights,
            self.in_blossom,
            self.label,
            self.dual,
            self.active,
            self.ids,
            self.parent,
            self.n,
            self.integer_weights,
            self.infinity,
        )
        if kind in (1, 2):
            edge_id = index
            a, b = self.edges[edge_id].unbind()
            v, w = (int(a), int(b)) if self.label[self.in_blossom[a]] == 1 else (int(b), int(a))
            self.allowed[edge_id] = True
            self.queued[v] = True
        elif kind == 3:
            self.expand_blossom(index, False)
        return kind

    def solve(self) -> Tensor:
        while True:
            self.label.zero_()
            self.label_edge.fill_(-1)
            self.allowed.zero_()
            self.queued.zero_()
            free = self.vertices[self.mate < 0]
            roots = torch.unique(self.in_blossom[free])
            self.label[free] = self.label[roots] = 1
            self.queued |= self.label[self.in_blossom] == 1
            augmented = False
            while True:
                v, w = _blossom_search(
                    self.neighbors,
                    self.adjacent_edges,
                    self.offsets,
                    self.weights,
                    self.in_blossom,
                    self.label,
                    self.label_edge,
                    self.base,
                    self.mate,
                    self.dual,
                    self.allowed,
                    self.queued,
                    self.vertices,
                    self.children,
                )
                if v >= 0:
                    base = self.scan_blossom(v, w)
                    if base >= 0:
                        self.add_blossom(base, v, w)
                    else:
                        self.augment_matching(v, w)
                        augmented = True
                        break
                elif self.update_duals() == 0:
                    break
            if not augmented:
                break
            expandable = self.active & (self.ids >= self.n) & (self.parent < 0) & (self.label == 1) & (self.dual == 0)
            for blossom_tensor in torch.nonzero(expandable, as_tuple=False).flatten():
                blossom = int(blossom_tensor)
                if self.active[blossom] and self.parent[blossom] < 0 and self.dual[blossom] == 0:
                    self.expand_blossom(blossom, True)
        return self.mate


def _blossom_leaves(blossom: int, n: int, vertices: Tensor, children: list[Tensor]) -> Tensor:
    """Expand only the requested subtree; keep no quadratic membership table."""
    if blossom < n:
        return vertices[blossom : blossom + 1]
    stack = [blossom]
    leaves = torch.jit.annotate(list[Tensor], [])
    while stack:
        current = stack.pop()
        links = children[current]
        leaves.append(links[links < n])
        for child in links[links >= n]:
            stack.append(int(child))
    return torch.cat(leaves)


def _blossom_search(
    neighbors: Tensor,
    adjacent_edges: Tensor,
    offsets: Tensor,
    weights: Tensor,
    in_blossom: Tensor,
    label: Tensor,
    label_edge: Tensor,
    base: Tensor,
    mate: Tensor,
    dual: Tensor,
    allowed: Tensor,
    queued: Tensor,
    vertices: Tensor,
    children: list[Tensor],
) -> tuple[int, int]:
    """Scan sparse neighbors until contraction or augmentation needs a tree walk."""
    n = vertices.numel()
    while bool(queued.any()):
        v = int(torch.nonzero(queued)[-1, 0])
        queued[v] = False
        start, end = int(offsets[v]), int(offsets[v + 1])
        adjacent = neighbors[start:end]
        edge_ids = adjacent_edges[start:end]
        different = in_blossom[adjacent] != in_blossom[v]
        tight = different & (dual[v] + dual[adjacent] - 2 * weights[edge_ids] <= 0)
        allowed[edge_ids] |= tight
        candidates = adjacent[different & allowed[edge_ids]]
        for neighbor in candidates:
            w = int(neighbor)
            v_blossom, w_blossom = int(in_blossom[v]), int(in_blossom[w])
            if v_blossom == w_blossom:
                continue
            if int(label[w_blossom]) == 0:
                label[w] = 2
                label[w_blossom] = 2
                label_edge[w, 0] = v
                label_edge[w_blossom, 0] = v
                label_edge[w, 1] = w
                label_edge[w_blossom, 1] = w
                root = int(base[w_blossom])
                matched = int(mate[root])
                outer = int(in_blossom[matched])
                label[matched] = 1
                label[outer] = 1
                label_edge[matched, 0] = root
                label_edge[outer, 0] = root
                label_edge[matched, 1] = matched
                label_edge[outer, 1] = matched
                queued[_blossom_leaves(outer, n, vertices, children)] = True
            elif int(label[w_blossom]) == 1:
                # Contraction can change the remaining candidates. Revisit v
                # using its current top-level blossom after the tree update.
                queued[v] = True
                return v, w
            elif int(label[w]) == 0:
                label[w] = 2
                label_edge[w, 0] = v
                label_edge[w, 1] = w
    return -1, -1


def _blossom_update_duals(
    edges: Tensor,
    weights: Tensor,
    in_blossom: Tensor,
    label: Tensor,
    dual: Tensor,
    active: Tensor,
    ids: Tensor,
    parent: Tensor,
    n: int,
    integer_weights: bool,
    infinity: Tensor,
) -> tuple[int, int]:
    """Compute the next primal-dual event without per-operation Python dispatch."""
    source, target = edges.unbind(1)
    source_blossom, target_blossom = in_blossom[source], in_blossom[target]
    source_label, target_label = label[source_blossom], label[target_blossom]
    slack = dual[source] + dual[target] - 2 * weights
    different = source_blossom != target_blossom
    unlabeled = different & (((source_label == 1) & (target_label == 0)) | ((target_label == 1) & (source_label == 0)))
    outer = different & (source_label == 1) & (target_label == 1)
    delta2, edge2 = torch.where(unlabeled, slack, infinity).min(dim=0)
    half_slack = torch.div(slack, 2, rounding_mode="floor") if integer_weights else slack / 2
    delta3, edge3 = torch.where(outer, half_slack, infinity).min(dim=0)
    top_inner = active & (ids >= n) & (parent < 0) & (label == 2)
    delta4, blossom4 = torch.where(top_inner, dual, infinity).min(dim=0)
    values = torch.stack((dual[:n].amin(), delta2, delta3, delta4))
    kind = int(values.argmin())
    delta = values[kind]
    vertex_labels = label[in_blossom]
    dual[:n] += torch.where(vertex_labels == 2, delta, torch.where(vertex_labels == 1, -delta, 0))
    top = active & (ids >= n) & (parent < 0)
    dual[n:] += torch.where(
        top[n:] & (label[n:] == 1),
        delta,
        torch.where(top[n:] & (label[n:] == 2), -delta, 0),
    )
    index = -1
    if kind == 1:
        index = int(edge2)
    elif kind == 2:
        index = int(edge3)
    elif kind == 3:
        index = int(blossom4)
    return kind, index


# Python 3.14+ uses the same routines eagerly: TorchScript does not support it.
if sys.version_info < (3, 14):
    _blossom_leaves = torch.jit.script(_blossom_leaves)
    _blossom_search = torch.jit.script(_blossom_search)
    _blossom_update_duals = torch.jit.script(_blossom_update_duals)
