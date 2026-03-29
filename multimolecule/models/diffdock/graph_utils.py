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

"""Pure-PyTorch replacements for torch_cluster and torch_scatter operations.

These functions replace:
- torch_cluster.radius_graph -> radius_graph
- torch_cluster.radius -> radius
- torch_cluster.knn_graph -> knn_graph
- torch_scatter.scatter -> scatter
- torch_scatter.scatter_mean -> scatter_mean
"""

from __future__ import annotations

import torch
from torch import Tensor


def radius_graph(
    pos: Tensor,
    r: float,
    batch: Tensor | None = None,
    max_num_neighbors: int = 32,
    loop: bool = False,
) -> Tensor:
    """Build a graph connecting all points within radius r.

    Replaces torch_cluster.radius_graph.

    Args:
        pos: Node positions, shape (N, D).
        r: Cutoff radius.
        batch: Batch vector assigning each node to a graph, shape (N,).
        max_num_neighbors: Maximum number of neighbors per node.
        loop: Whether to include self-loops.

    Returns:
        Edge index tensor of shape (2, E).
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    # Process per-graph to respect batch boundaries
    edge_indices = []
    for graph_id in batch.unique():
        mask = batch == graph_id
        idx = mask.nonzero(as_tuple=True)[0]
        local_pos = pos[idx]
        n = local_pos.size(0)

        # Compute pairwise distances
        dist = torch.cdist(local_pos, local_pos)  # (n, n)

        # Mask: within radius and not self (unless loop)
        valid = dist < r
        if not loop:
            valid.fill_diagonal_(False)

        # Limit neighbors: for each node, keep at most max_num_neighbors
        if max_num_neighbors < n:
            # Set invalid entries to inf, then topk
            dist_masked = dist.clone()
            dist_masked[~valid] = float("inf")
            _, topk_idx = dist_masked.topk(min(max_num_neighbors, n), dim=1, largest=False)
            # Rebuild valid mask from topk
            valid_topk = torch.zeros_like(valid)
            src_idx = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(topk_idx)
            valid_topk[src_idx, topk_idx] = True
            valid = valid & valid_topk

        src, dst = valid.nonzero(as_tuple=True)
        # Map back to global indices
        edge_indices.append(torch.stack([idx[src], idx[dst]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return torch.zeros(2, 0, dtype=torch.long, device=pos.device)


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    max_num_neighbors: int = 10000,
) -> Tensor:
    """Find all pairs (i, j) where ||x[j] - y[i]|| < r, across batch boundaries.

    Replaces torch_cluster.radius. Returns edges from y to x.

    Args:
        x: Target positions, shape (N, D).
        y: Source positions, shape (M, D).
        r: Cutoff radius (scalar or per-graph if tensor).
        batch_x: Batch vector for x, shape (N,).
        batch_y: Batch vector for y, shape (M,).
        max_num_neighbors: Maximum neighbors per source node.

    Returns:
        Edge index tensor of shape (2, E) where [0] indexes into y and [1] indexes into x.
    """
    if batch_x is None:
        batch_x = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(y.size(0), dtype=torch.long, device=y.device)

    edge_indices = []
    for graph_id in batch_x.unique():
        mask_x = batch_x == graph_id
        mask_y = batch_y == graph_id
        idx_x = mask_x.nonzero(as_tuple=True)[0]
        idx_y = mask_y.nonzero(as_tuple=True)[0]

        if idx_x.size(0) == 0 or idx_y.size(0) == 0:
            continue

        dist = torch.cdist(y[idx_y], x[idx_x])  # (M_g, N_g)
        valid = dist < r

        # Limit neighbors per source node
        if max_num_neighbors < idx_x.size(0):
            dist_masked = dist.clone()
            dist_masked[~valid] = float("inf")
            k = min(max_num_neighbors, idx_x.size(0))
            _, topk_idx = dist_masked.topk(k, dim=1, largest=False)
            valid_topk = torch.zeros_like(valid)
            src_idx = torch.arange(idx_y.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)
            valid_topk[src_idx, topk_idx] = True
            valid = valid & valid_topk

        src_local, dst_local = valid.nonzero(as_tuple=True)
        edge_indices.append(torch.stack([idx_y[src_local], idx_x[dst_local]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return torch.zeros(2, 0, dtype=torch.long, device=x.device)


def knn_graph(
    pos: Tensor,
    k: int,
    batch: Tensor | None = None,
    loop: bool = False,
) -> Tensor:
    """Build a k-nearest-neighbor graph.

    Replaces torch_cluster.knn_graph.

    Args:
        pos: Node positions, shape (N, D).
        k: Number of neighbors.
        batch: Batch vector, shape (N,).
        loop: Whether to include self-loops.

    Returns:
        Edge index tensor of shape (2, E).
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_indices = []
    for graph_id in batch.unique():
        mask = batch == graph_id
        idx = mask.nonzero(as_tuple=True)[0]
        local_pos = pos[idx]
        n = local_pos.size(0)

        dist = torch.cdist(local_pos, local_pos)
        if not loop:
            dist.fill_diagonal_(float("inf"))

        actual_k = min(k, n - (0 if loop else 1))
        if actual_k <= 0:
            continue

        _, topk_idx = dist.topk(actual_k, dim=1, largest=False)
        src = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(topk_idx).reshape(-1)
        dst = topk_idx.reshape(-1)

        edge_indices.append(torch.stack([idx[src], idx[dst]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return torch.zeros(2, 0, dtype=torch.long, device=pos.device)


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> Tensor:
    """Scatter operation: aggregate src values by index.

    Replaces torch_scatter.scatter.

    Args:
        src: Source tensor.
        index: Index tensor, same size as src along dim.
        dim: Dimension along which to scatter.
        dim_size: Size of the output along dim.
        reduce: Aggregation method ('sum', 'mean', 'min', 'max').

    Returns:
        Aggregated tensor.
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # Expand index to match src shape
    idx = index
    if src.dim() > 1 and idx.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        idx = idx.view(shape).expand_as(src)

    out = torch.zeros(*[dim_size if d == dim else s for d, s in enumerate(src.shape)],
                       dtype=src.dtype, device=src.device)

    if reduce == "sum":
        out.scatter_add_(dim, idx, src)
    elif reduce == "mean":
        out.scatter_add_(dim, idx, src)
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        count = count.clamp(min=1)
        # Reshape count for broadcasting
        shape = [1] * out.dim()
        shape[dim] = -1
        out = out / count.view(shape)
    elif reduce in ("min", "max"):
        fill = float("inf") if reduce == "min" else float("-inf")
        out.fill_(fill)
        out.scatter_reduce_(dim, idx, src, reduce=reduce, include_self=False)
        # Replace unfilled positions with 0
        out[out == fill] = 0
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")

    return out


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Scatter mean: average src values by index.

    Replaces torch_scatter.scatter_mean.
    """
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
