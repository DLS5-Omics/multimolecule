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

"""Spatial graph construction from 3D coordinates.

Pure-PyTorch replacements for torch_cluster operations:
- radius_graph: connect all points within a distance cutoff
- radius: cross-graph radius search between two point sets
- knn_graph: k-nearest-neighbor graph construction
"""

from __future__ import annotations

import torch
from torch import Tensor


def _empty_edge_index(device: torch.device) -> Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _validate_batch(batch: Tensor | None, size: int, device: torch.device, name: str) -> Tensor:
    if batch is None:
        return torch.zeros(size, dtype=torch.long, device=device)
    if batch.dim() != 1 or batch.numel() != size:
        raise ValueError(f"{name} must be a 1D tensor with length {size}, but got shape {tuple(batch.shape)}")
    return batch.to(device=device, dtype=torch.long)


def _validate_positions(pos: Tensor, name: str) -> None:
    if pos.dim() != 2:
        raise ValueError(f"{name} must have shape (N, D), but got shape {tuple(pos.shape)}")
    if pos.numel() and not bool(torch.isfinite(pos).all().item()):
        raise ValueError(f"{name} must contain finite coordinates")


def radius_graph(
    pos: Tensor,
    r: float,
    batch: Tensor | None = None,
    max_num_neighbors: int = 32,
    loop: bool = False,
) -> Tensor:
    """Build a graph connecting all points within radius r.

    Args:
        pos: Node positions, shape (N, D).
        r: Cutoff radius.
        batch: Batch vector assigning each node to a graph, shape (N,).
        max_num_neighbors: Maximum number of neighbors per node.
        loop: Whether to include self-loops.

    Returns:
        Edge index tensor of shape (2, E).
    """
    _validate_positions(pos, "pos")
    if r < 0:
        raise ValueError(f"r must be non-negative, but got {r}")
    if max_num_neighbors < 0:
        raise ValueError(f"max_num_neighbors must be non-negative, but got {max_num_neighbors}")
    batch = _validate_batch(batch, pos.size(0), pos.device, "batch")

    edge_indices = []
    for graph_id in batch.unique():
        mask = batch == graph_id
        idx = mask.nonzero(as_tuple=True)[0]
        local_pos = pos[idx]
        n = local_pos.size(0)

        dist = torch.cdist(local_pos, local_pos)
        valid = dist < r
        if not loop:
            valid.fill_diagonal_(False)

        if max_num_neighbors < n:
            dist_masked = dist.clone()
            dist_masked[~valid] = float("inf")
            _, topk_idx = dist_masked.topk(min(max_num_neighbors, n), dim=1, largest=False)
            valid_topk = torch.zeros_like(valid)
            target_idx = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(topk_idx)
            valid_topk[target_idx, topk_idx] = True
            valid = valid & valid_topk

        target, source = valid.nonzero(as_tuple=True)
        edge_indices.append(torch.stack([idx[source], idx[target]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return _empty_edge_index(pos.device)


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    max_num_neighbors: int = 10000,
) -> Tensor:
    """Find all pairs (i, j) where ||x[j] - y[i]|| < r across batch boundaries.

    Returns edges from y to x: result[0] indexes into y, result[1] indexes into x.

    Args:
        x: Target positions, shape (N, D).
        y: Source positions, shape (M, D).
        r: Cutoff radius.
        batch_x: Batch vector for x, shape (N,).
        batch_y: Batch vector for y, shape (M,).
        max_num_neighbors: Maximum neighbors per source node.

    Returns:
        Edge index tensor of shape (2, E).
    """
    _validate_positions(x, "x")
    _validate_positions(y, "y")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.size(-1) != y.size(-1):
        raise ValueError(f"x and y must have the same feature dimension, got {x.size(-1)} and {y.size(-1)}")
    if r < 0:
        raise ValueError(f"r must be non-negative, but got {r}")
    if max_num_neighbors < 0:
        raise ValueError(f"max_num_neighbors must be non-negative, but got {max_num_neighbors}")
    batch_x = _validate_batch(batch_x, x.size(0), x.device, "batch_x")
    batch_y = _validate_batch(batch_y, y.size(0), y.device, "batch_y")

    edge_indices = []
    for graph_id in batch_x.unique():
        mask_x = batch_x == graph_id
        mask_y = batch_y == graph_id
        idx_x = mask_x.nonzero(as_tuple=True)[0]
        idx_y = mask_y.nonzero(as_tuple=True)[0]

        if idx_x.size(0) == 0 or idx_y.size(0) == 0:
            continue

        dist = torch.cdist(y[idx_y], x[idx_x])
        valid = dist < r

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
    return _empty_edge_index(x.device)


def knn_graph(
    pos: Tensor,
    k: int,
    batch: Tensor | None = None,
    loop: bool = False,
) -> Tensor:
    """Build a k-nearest-neighbor graph.

    Args:
        pos: Node positions, shape (N, D).
        k: Number of neighbors.
        batch: Batch vector, shape (N,).
        loop: Whether to include self-loops.

    Returns:
        Edge index tensor of shape (2, E).
    """
    _validate_positions(pos, "pos")
    if k < 0:
        raise ValueError(f"k must be non-negative, but got {k}")
    batch = _validate_batch(batch, pos.size(0), pos.device, "batch")

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
        target = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(topk_idx).reshape(-1)
        source = topk_idx.reshape(-1)

        edge_indices.append(torch.stack([idx[source], idx[target]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return _empty_edge_index(pos.device)
