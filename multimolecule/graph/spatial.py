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

Distance blocks stay within a fixed workspace budget. Half precision inputs use
float32 arithmetic. Sparse CPU radius searches in up to three dimensions use
neighboring grid cells and direct distances, with the same workspace budget;
dense, small, high-dimensional and numerically extreme inputs use distance blocks.
If max(abs(pos)) exceeds sqrt(finfo.max) / (8 * sqrt(max(D, 1))),
float32 arithmetic is promoted to direct float64 distances; exceeding that bound
in float64 raises ValueError, even for an otherwise empty result. MM distances
use a common translated origin only when no point's squared norm would increase.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import product
from numbers import Integral, Real

import torch
from torch import Tensor

# Bound the distance workspace independently of the total graph size. Output
# edges and the nearest-neighbor candidates are additional, unavoidable storage.
_DISTANCE_BLOCK_BYTES = 16 * 1024 * 1024

# Building a cell index is useful only once it replaces a substantial distance
# matrix. Very small groups keep the cheaper dense implementation below.
_GRID_MIN_PAIRS = 1024 * 1024


def _empty_edge_index(device: torch.device) -> Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _validate_batch(batch: Tensor | None, size: int, device: torch.device, name: str) -> Tensor | None:
    if batch is None:
        return None
    if batch.dim() != 1 or batch.numel() != size:
        raise ValueError(f"{name} must be a 1D tensor with length {size}, but got shape {tuple(batch.shape)}")
    if batch.dtype == torch.bool or batch.is_floating_point() or batch.is_complex():
        raise TypeError(f"{name} must use an integer dtype")
    return batch.to(device=device, dtype=torch.long)


def _validate_positions(pos: Tensor, name: str) -> tuple[torch.dtype, bool]:
    if pos.dim() != 2:
        raise ValueError(f"{name} must have shape (N, D), but got shape {tuple(pos.shape)}")
    if pos.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"{name} must use a supported floating-point dtype, but got {pos.dtype}")
    maximum = pos.abs().amax().item() if pos.numel() else 0.0
    if not math.isfinite(maximum):
        raise ValueError(f"{name} must contain finite coordinates")
    dtype = torch.float32 if pos.dtype in (torch.float16, torch.bfloat16) else pos.dtype
    # Leave room for subtraction, squared norms, and MM intermediates, including
    # the common-origin translation used below. Finite coordinates alone do not
    # imply that their squared distances fit in the input dtype.
    safe_magnitude = math.sqrt(torch.finfo(dtype).max) / (8 * math.sqrt(max(1, pos.size(1))))
    direct = maximum > safe_magnitude
    if direct:
        dtype = torch.float64
        safe_magnitude = math.sqrt(torch.finfo(dtype).max) / (8 * math.sqrt(max(1, pos.size(1))))
        if maximum > safe_magnitude:
            raise ValueError(f"{name} coordinates exceed the supported float64 distance range")
    return dtype, direct


def _validate_radius(r: float) -> float:
    if isinstance(r, bool) or not isinstance(r, Real):
        raise TypeError("r must be a real number")
    r = float(r)
    if math.isnan(r) or r < 0:
        raise ValueError(f"r must be non-negative and not NaN, but got {r}")
    return r


def _validate_neighbor_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, but got {value}")
    return int(value)


def _batch_groups(batch: Tensor | None, size: int, device: torch.device) -> dict[int, Tensor]:
    if not size:
        return {}
    if batch is None:
        return {0: torch.arange(size, device=device)}
    order = batch.argsort(stable=True)
    graph_ids, counts = torch.unique_consecutive(batch[order], return_counts=True)
    groups = {}
    offset = 0
    for graph_id, count in zip(graph_ids.tolist(), counts.tolist()):
        groups[graph_id] = order[offset : offset + count]
        offset += count
    return groups


def _distance_blocks(query: Tensor, reference: Tensor, direct: bool = False) -> tuple[int, int, str]:
    # Keep the cdist algorithm selected by the original, full point sets. The
    # default would otherwise switch at 25 points as chunk sizes change.
    mode = (
        "use_mm_for_euclid_dist"
        if not direct and (query.size(0) > 25 or reference.size(0) > 25)
        else "donot_use_mm_for_euclid_dist"
    )
    elements = max(1, _DISTANCE_BLOCK_BYTES // query.element_size())
    columns = min(reference.size(0), elements)
    rows = min(query.size(0), max(1, elements // columns))
    return rows, columns, mode


def _center_positions(query: Tensor, reference: Tensor, mode: str) -> tuple[Tensor, Tensor]:
    if mode != "use_mm_for_euclid_dist":
        return query, reference
    # Translation removes a common large offset before forming squared norms.
    # In particular, nearby float32 points around 1e6 must not all become zero
    # distance through cancellation of their uncentered squared norms.
    origin = reference[0]
    # A distant first point must not move a well-conditioned cluster near zero
    # far from zero. Apply the shift only when no point's squared norm grows:
    # ||point - origin||² <= ||point||² iff point·origin >= ||origin||² / 2.
    half_norm = origin.square().sum() * 0.5
    use_center = ((query * origin).sum(dim=1) >= half_norm).all()
    if query is not reference:
        use_center = use_center & ((reference * origin).sum(dim=1) >= half_norm).all()
    centered_query = torch.where(use_center, query - origin, query)
    centered_reference = (
        centered_query if query is reference else torch.where(use_center, reference - origin, reference)
    )
    return centered_query, centered_reference


def _set_diagonal(dist: Tensor, row_start: int, column_start: int, value: float) -> None:
    if row_start == column_start and dist.size(0) == dist.size(1):
        dist.fill_diagonal_(value)
        return
    start = max(row_start, column_start)
    stop = min(row_start + dist.size(0), column_start + dist.size(1))
    if start >= stop:
        return
    diagonal = torch.arange(start, stop, device=dist.device)
    dist[diagonal - row_start, diagonal - column_start] = value


def _nearest_blocks(
    query: Tensor, reference: Tensor, k: int, exclude_diagonal: bool = False, direct: bool = False
) -> Iterator[tuple[int, Tensor, Tensor]]:
    self_search = query is reference
    rows, columns, mode = _distance_blocks(query, reference, direct)
    query, reference = _center_positions(query, reference, mode)
    for row_start in range(0, query.size(0), rows):
        local_query = query[row_start : row_start + rows]
        best_dist = best_index = None
        for column_start in range(0, reference.size(0), columns):
            dist = torch.cdist(local_query, reference[column_start : column_start + columns], compute_mode=mode)
            if self_search:
                _set_diagonal(dist, row_start, column_start, float("inf") if exclude_diagonal else 0.0)
            local_dist, local_index = dist.topk(min(k, dist.size(1)), dim=1, largest=False)
            del dist
            if column_start:
                local_index = local_index + column_start
            if best_dist is None:
                best_dist, best_index = local_dist, local_index
            else:
                assert best_index is not None
                candidates = torch.cat((best_dist, local_dist), dim=1)
                indices = torch.cat((best_index, local_index), dim=1)
                best_dist, selection = candidates.topk(min(k, candidates.size(1)), dim=1, largest=False)
                best_index = indices.gather(1, selection)
        assert best_dist is not None and best_index is not None
        yield row_start, best_dist, best_index


def _grid_radius_pairs(
    query: Tensor, reference: Tensor, r: float, k: int, exclude_diagonal: bool, direct: bool
) -> Tensor | None:
    """Use neighboring CPU grid cells, or return None for the dense fallback.

    Points in one query cell share the same candidate cells. Batched cdist
    therefore compares each occupied query cell with its local candidates,
    rather than constructing a ragged candidate list for every point.
    """
    dimension = query.size(1)
    if (
        query.device.type != "cpu"
        or dimension not in (1, 2, 3)
        or query.size(0) * reference.size(0) < _GRID_MIN_PAIRS
        or direct
        or not math.isfinite(r)
    ):
        return None

    # Enlarge the cells slightly and limit the grid's dynamic range. The margin
    # dominates float64 division/subtraction rounding at this range, so points
    # strictly closer than r cannot skip an adjacent cell at a boundary.
    width = r * (1 + 2**-20)
    if not math.isfinite(width) or width < torch.finfo(torch.float64).tiny:
        return None
    origin = torch.minimum(query.amin(dim=0), reference.amin(dim=0)).double()
    query_cells = (query.double() - origin) / width
    reference_cells = (reference.double() - origin) / width
    extent = torch.maximum(query_cells.amax(dim=0), reference_cells.amax(dim=0))
    if not bool((extent <= 2**20).all()):
        return None
    query_cells = query_cells.floor().long() + 1
    reference_cells = reference_cells.floor().long() + 1
    shape = extent.floor().long() + 3
    strides = torch.ones(dimension, dtype=torch.long)
    if dimension > 1:
        strides[:-1] = shape[1:].flip(0).cumprod(0).flip(0)
    query_keys = (query_cells * strides).sum(dim=1)
    reference_keys, reference_order = (reference_cells * strides).sum(dim=1).sort(stable=True)
    query_keys, query_order = query_keys.sort(stable=True)
    _, query_counts = torch.unique_consecutive(query_keys, return_counts=True)
    query_starts = query_counts.cumsum(0) - query_counts
    cells = query_cells[query_order[query_starts]]
    offsets = torch.tensor(tuple(product((-1, 0, 1), repeat=dimension)), dtype=torch.long)
    neighbor_keys = ((cells[:, None] + offsets) * strides).sum(dim=2)
    left = torch.searchsorted(reference_keys, neighbor_keys)
    counts = torch.searchsorted(reference_keys, neighbor_keys, right=True) - left
    reference_counts = counts.sum(dim=1)
    # Dense or poorly distributed neighborhoods do not repay the grid setup and
    # padding. Avoid both wasted work and an oversized single-cell workspace.
    candidate_pairs = (query_counts * reference_counts).sum().item()
    if candidate_pairs > query.size(0) * reference.size(0) // 4:
        return None
    max_query = int(query_counts.max())
    max_reference = int(reference_counts.max())
    if not max_reference:
        return _empty_edge_index(query.device)
    # Conservatively estimate explicit block tensors: cdist and masks, topk's
    # distances/indices plus gathered sources, nonzero's three index arrays,
    # and the two output gathers. Final edge storage and the O(3**D * N) grid
    # index are separate. Operator-internal allocator workspace is not counted;
    # this estimate does not bound total process memory.
    selected_width = min(k, max_reference)
    selection_bytes = query.element_size() + 2 * 8 if k < max_reference else 0
    bytes_per_cell = max_query * max_reference * (query.element_size() + 3) + (max_query + max_reference) * (
        3 * dimension * query.element_size() + 6 * 8
    )
    bytes_per_cell += max_query * selected_width * (selection_bytes + 5 * 8)
    bytes_per_cell += 2 * offsets.size(0) * 8
    cells_per_block = _DISTANCE_BLOCK_BYTES // bytes_per_cell
    if not cells_per_block:
        return None

    self_search = query is reference
    query_rank = torch.arange(max_query)
    reference_rank = torch.arange(max_reference)
    pairs = []
    for start in range(0, query_counts.numel(), cells_per_block):
        stop = start + cells_per_block
        local_query_counts = query_counts[start:stop]
        local_reference_counts = reference_counts[start:stop]
        query_index = query_starts[start:stop, None] + query_rank
        query_index = query_order[query_index.clamp_max(query.size(0) - 1)]
        local_counts = counts[start:stop]
        cumulative = local_counts.cumsum(dim=1)
        ranks = reference_rank.expand(local_counts.size(0), -1).contiguous()
        cell = torch.searchsorted(cumulative, ranks, right=True).clamp_max(offsets.size(0) - 1)
        before = (cumulative - local_counts).gather(1, cell)
        reference_index = left[start:stop].gather(1, cell) + ranks - before
        reference_index = reference_order[reference_index.clamp_max(reference.size(0) - 1)]
        # Candidate dimensions are at most three. Direct differences avoid MM
        # cancellation changing the strict cutoff near a cell boundary.
        dist = torch.cdist(query[query_index], reference[reference_index], compute_mode="donot_use_mm_for_euclid_dist")
        query_valid = query_rank[None] < local_query_counts[:, None]
        reference_valid = reference_rank[None] < local_reference_counts[:, None]
        dist.masked_fill_(~reference_valid[:, None], float("inf"))
        if self_search:
            diagonal = query_index[:, :, None] == reference_index[:, None, :]
            diagonal &= reference_valid[:, None]
            dist.masked_fill_(diagonal, float("inf") if exclude_diagonal else 0.0)
        if k < max_reference:
            dist, selected = dist.topk(k, dim=2, largest=False)
            source = reference_index[:, None].expand(-1, max_query, -1).gather(2, selected)
        else:
            source = reference_index[:, None].expand(-1, max_query, -1)
        cell_index, row, column = ((dist < r) & query_valid[:, :, None]).nonzero(as_tuple=True)
        pairs.append(torch.stack((query_index[cell_index, row], source[cell_index, row, column])))
        # Release the large temporaries before allocating the next block.
        del dist, source, cell_index, row, column
        if k < max_reference:
            del selected
        if self_search:
            del diagonal
    result = torch.cat(pairs, dim=1)
    # Radius results keep the established query-major, source-index ordering.
    return result[:, (result[0] * reference.size(0) + result[1]).argsort()]


def _radius_pairs(
    query: Tensor, reference: Tensor, r: float, k: int, exclude_diagonal: bool = False, direct: bool = False
) -> Tensor:
    indexed = _grid_radius_pairs(query, reference, r, k, exclude_diagonal, direct)
    if indexed is not None:
        return indexed
    pairs = []
    if k < reference.size(0):
        for row_start, dist, index in _nearest_blocks(query, reference, k, exclude_diagonal, direct):
            # radius historically emits neighbors in original-index order,
            # whereas knn emits them in distance order.
            index, order = index.sort(dim=1)
            valid = dist.gather(1, order) < r
            row, column = valid.nonzero(as_tuple=True)
            pairs.append(torch.stack((row + row_start, index[row, column])))
    else:
        self_search = query is reference
        rows, columns, mode = _distance_blocks(query, reference, direct)
        query, reference = _center_positions(query, reference, mode)
        for row_start in range(0, query.size(0), rows):
            row_pairs = []
            for column_start in range(0, reference.size(0), columns):
                dist = torch.cdist(
                    query[row_start : row_start + rows],
                    reference[column_start : column_start + columns],
                    compute_mode=mode,
                )
                if self_search:
                    _set_diagonal(dist, row_start, column_start, float("inf") if exclude_diagonal else 0.0)
                row, column = (dist < r).nonzero(as_tuple=True)
                row_pairs.append(torch.stack((row + row_start, column + column_start)))
                del dist
            local_pairs = torch.cat(row_pairs, dim=1)
            if columns < reference.size(0):
                order = (local_pairs[0] * reference.size(0) + local_pairs[1]).argsort()
                local_pairs = local_pairs[:, order]
            pairs.append(local_pairs)
    return torch.cat(pairs, dim=1)


@torch.no_grad()
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

    Ties at the neighbor limit have no guaranteed ordering.

    ``r=+inf`` includes all finite distances, subject to the neighbor limit.
    NaN radii, non-integer batch IDs, and non-integer neighbor limits are invalid.
    Half precision coordinates use float32 distances. Coordinates too large for
    safe float32 squared distances use direct float64 distances; coordinates
    beyond the safe float64 range raise ``ValueError`` rather than overflowing.
    Large sparse CPU inputs with one to three features use a grid and direct
    Euclidean distances. Near the cutoff, these can differ from the rounded
    matrix-multiplication distances used by the dense fallback.
    """
    distance_dtype, direct = _validate_positions(pos, "pos")
    r = _validate_radius(r)
    max_num_neighbors = _validate_neighbor_count(max_num_neighbors, "max_num_neighbors")
    batch = _validate_batch(batch, pos.size(0), pos.device, "batch")
    if not pos.size(0) or max_num_neighbors == 0 or r == 0:
        return _empty_edge_index(pos.device)

    pos = pos.to(dtype=distance_dtype)
    edge_indices = []
    for idx in _batch_groups(batch, pos.size(0), pos.device).values():
        local_pos = pos[idx]
        target, source = _radius_pairs(local_pos, local_pos, r, max_num_neighbors, not loop, direct)
        edge_indices.append(torch.stack([idx[source], idx[target]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return _empty_edge_index(pos.device)


@torch.no_grad()
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

    Ties at the neighbor limit have no guaranteed ordering.

    ``r=+inf`` includes all finite distances, subject to the neighbor limit.
    Both inputs must use the same floating-point dtype. Distance precision and
    range validation follow :func:`radius_graph`, including ``ValueError`` for
    coordinates beyond the safe float64 distance range. Sparse CPU grid searches
    also use direct distances as described there.
    """
    dtype_x, direct_x = _validate_positions(x, "x")
    dtype_y, direct_y = _validate_positions(y, "y")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.size(-1) != y.size(-1):
        raise ValueError(f"x and y must have the same feature dimension, got {x.size(-1)} and {y.size(-1)}")
    if x.dtype != y.dtype:
        raise TypeError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    r = _validate_radius(r)
    max_num_neighbors = _validate_neighbor_count(max_num_neighbors, "max_num_neighbors")
    batch_x = _validate_batch(batch_x, x.size(0), x.device, "batch_x")
    batch_y = _validate_batch(batch_y, y.size(0), y.device, "batch_y")
    if not x.size(0) or not y.size(0) or max_num_neighbors == 0 or r == 0:
        return _empty_edge_index(x.device)

    distance_dtype = torch.promote_types(dtype_x, dtype_y)
    x, y = x.to(dtype=distance_dtype), y.to(dtype=distance_dtype)
    groups_x = _batch_groups(batch_x, x.size(0), x.device)
    groups_y = _batch_groups(batch_y, y.size(0), y.device)
    edge_indices = []
    for graph_id, idx_x in groups_x.items():
        idx_y = groups_y.get(graph_id)
        if idx_y is None:
            continue
        src_local, dst_local = _radius_pairs(y[idx_y], x[idx_x], r, max_num_neighbors, direct=direct_x or direct_y)
        edge_indices.append(torch.stack([idx_y[src_local], idx_x[dst_local]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return _empty_edge_index(x.device)


@torch.no_grad()
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

    Ties between equally distant neighbors have no guaranteed ordering.

    Coordinates and batch IDs follow :func:`radius_graph` validation. Large
    float32 coordinates use direct float64 distances; coordinates beyond the
    safe float64 distance range raise ``ValueError``.
    """
    distance_dtype, direct = _validate_positions(pos, "pos")
    k = _validate_neighbor_count(k, "k")
    batch = _validate_batch(batch, pos.size(0), pos.device, "batch")
    if not pos.size(0) or k == 0:
        return _empty_edge_index(pos.device)

    pos = pos.to(dtype=distance_dtype)
    edge_indices = []
    for idx in _batch_groups(batch, pos.size(0), pos.device).values():
        local_pos = pos[idx]
        n = local_pos.size(0)
        actual_k = min(k, n - (0 if loop else 1))
        if actual_k <= 0:
            continue
        for row_start, _, topk_idx in _nearest_blocks(local_pos, local_pos, actual_k, not loop, direct):
            target = torch.arange(row_start, row_start + topk_idx.size(0), device=pos.device)
            target = target.unsqueeze(1).expand_as(topk_idx).reshape(-1)
            source = topk_idx.reshape(-1)
            edge_indices.append(torch.stack([idx[source], idx[target]], dim=0))

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    return _empty_edge_index(pos.device)
