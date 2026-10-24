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
from math import prod

import torch
from torch import Tensor


def linear_sum_assignment(cost_matrix: Tensor, maximize: bool = False) -> Tensor:
    """Find an optimal assignment covering the smaller matrix dimension.

    Args:
        cost_matrix: Real tensor of shape ``(*batch, rows, columns)``.
            Minimization permits ``+inf`` to forbid individual assignments;
            maximization permits ``-inf``. NaN and the opposite infinity are
            invalid. Each batch member must admit a full assignment.
        maximize: Maximize the sum of selected entries instead of minimizing.

    Returns:
        A long tensor of shape ``(*batch, min(rows, columns), 2)`` on the input
        device, containing ``(row, column)`` pairs sorted by the original row.

    Floating inputs use float64 primal-dual arithmetic. Integer inputs retain
    exact int64 arithmetic, including differences above ``2**53``. To keep
    potential updates within their arithmetic range, finite entry magnitudes
    may not exceed ``dtype_max / (8 * (min(rows, columns) + 1))``, where
    ``dtype_max`` is the int64 or float64 maximum, respectively.

    The shortest augmenting path Hungarian algorithm batches column scans and
    updates potentials once per augmentation. CUDA search and augmentation
    loops use shape-derived bounds and active masks. Validation and the final
    feasibility check may synchronize. CPU searches may exit early. No graph data leaves its device.
    Input gradients are not tracked because the result consists of indices.

    Raises:
        TypeError: The input is not a real numeric tensor, or ``maximize`` is
            not a bool.
        ValueError: The shape, values, arithmetic range, or feasibility is
            invalid. A failure in any batch member rejects the whole batch.
    """
    if not isinstance(cost_matrix, Tensor):
        raise TypeError("cost_matrix must be a torch tensor.")
    if cost_matrix.layout != torch.strided:
        raise TypeError("cost_matrix must be a dense strided tensor.")
    if cost_matrix.ndim < 2:
        raise ValueError("cost_matrix must have at least two dimensions.")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    integer_costs = cost_matrix.dtype in integer_dtypes
    if not integer_costs and not cost_matrix.is_floating_point():
        raise TypeError("cost_matrix must contain real floating-point or integer values.")

    batch_shape = cost_matrix.shape[:-2]
    rows, columns = cost_matrix.shape[-2:]
    size = min(rows, columns)
    batch_size = prod(batch_shape)
    if size == 0 or batch_size == 0:
        return torch.empty((*batch_shape, size, 2), dtype=torch.long, device=cost_matrix.device)

    costs = cost_matrix.detach().to(torch.int64 if integer_costs else torch.float64)
    costs = costs.reshape(batch_size, rows, columns)
    transposed = rows > columns
    if transposed:
        costs = costs.transpose(1, 2)

    sentinel: int | float
    limit: int | float
    if integer_costs:
        sentinel = torch.iinfo(torch.int64).max
        limit = sentinel // (8 * (size + 1))
        if ((costs < -limit) | (costs > limit)).any():
            raise ValueError("integer cost_matrix entries exceed the exact int64 assignment arithmetic range.")
    else:
        sentinel = float("inf")
        limit = torch.finfo(torch.float64).max / (8 * (size + 1))
        invalid_infinity = torch.isposinf(costs) if maximize else torch.isneginf(costs)
        if (torch.isnan(costs) | invalid_infinity).any():
            raise ValueError("cost_matrix contains NaN or an infinity incompatible with the optimization direction.")
        if (torch.isfinite(costs) & ((costs < -limit) | (costs > limit))).any():
            raise ValueError("floating cost_matrix entries exceed the float64 assignment arithmetic range.")
    if maximize:
        costs = -costs

    column_for_row, failed = _assignment_core(costs, integer_costs, costs.new_tensor(sentinel))
    if failed.any():
        raise ValueError("cost_matrix is infeasible: no full assignment covers the smaller dimension.")

    row_indices = torch.arange(size, device=costs.device).expand(batch_size, -1)
    if transposed:
        order = column_for_row.argsort(dim=1)
        pairs = torch.stack((column_for_row.gather(1, order), row_indices.gather(1, order)), dim=2)
    else:
        pairs = torch.stack((row_indices, column_for_row), dim=2)
    return pairs.reshape(*batch_shape, size, 2)


def _assignment_core(costs: Tensor, integer_costs: bool, sentinel: Tensor) -> tuple[Tensor, Tensor]:
    """Run batched tensor assignment with reusable work arrays."""
    batch_size, size, width = costs.shape
    if costs.device.type == "cpu" and batch_size == 1:
        return _assignment_single_cpu(costs, integer_costs, sentinel)
    u, v = costs.new_zeros((batch_size, size)), costs.new_zeros((batch_size, width))
    row_for_column = torch.full((batch_size, width), -1, dtype=torch.long, device=costs.device)
    column_for_row = row_for_column.new_full((batch_size, size), -1)
    predecessor = torch.zeros_like(row_for_column)
    visited = torch.zeros_like(row_for_column, dtype=torch.bool)
    distance = costs.new_empty((batch_size, width))
    batch_indices = torch.arange(batch_size, device=costs.device)
    failed = torch.zeros(batch_size, dtype=torch.bool, device=costs.device)
    current_column = row_for_column.new_zeros(batch_size)
    cpu = costs.device.type == "cpu"
    for root in range(size):
        visited.zero_()
        distance.copy_(sentinel)
        current_row = row_for_column.new_full((batch_size,), root)
        minimum = costs.new_zeros(batch_size)
        active = ~failed
        # At most root occupied columns precede a free column.
        for _ in range(root + 1):
            reduced = (
                minimum.unsqueeze(1) + costs[batch_indices, current_row] - u.gather(1, current_row.unsqueeze(1)) - v
            )
            improved = active.unsqueeze(1) & ~visited & (reduced < distance)
            distance = torch.where(improved, reduced, distance)
            predecessor = torch.where(improved, current_row.unsqueeze(1), predecessor)
            next_minimum, next_column = torch.where(visited, sentinel, distance).min(dim=1)
            finite = next_minimum != sentinel if integer_costs else torch.isfinite(next_minimum)
            failed |= active & ~finite
            step = active & finite
            minimum = torch.where(step, next_minimum, minimum)
            current_column = torch.where(step, next_column, current_column)
            columns = current_column.unsqueeze(1)
            visited.scatter_(1, columns, visited.gather(1, columns) | step.unsqueeze(1))
            next_row = row_for_column.gather(1, columns).squeeze(1)
            active = step & (next_row >= 0)
            current_row = next_row.clamp_min(0)
            if cpu and not active.any():
                break
        failed |= active
        valid = ~failed
        # All shortest distances are absolute within this search. Apply their
        # cumulative dual changes only once, after reaching a free column.
        u[:, root] += torch.where(valid, minimum, 0)
        change = torch.where(visited & valid.unsqueeze(1), minimum.unsqueeze(1) - distance, 0)
        u.scatter_add_(1, row_for_column.clamp_min(0), torch.where(row_for_column >= 0, change, 0))
        v -= change
        augmenting = valid
        for _ in range(root + 1):
            columns = current_column.unsqueeze(1)
            row = predecessor.gather(1, columns)
            previous = column_for_row.gather(1, row).squeeze(1)
            row_for_column.scatter_(
                1, columns, torch.where(augmenting.unsqueeze(1), row, row_for_column.gather(1, columns))
            )
            column_for_row.scatter_(
                1, row, torch.where(augmenting.unsqueeze(1), columns, column_for_row.gather(1, row))
            )
            augmenting &= row.squeeze(1) != root
            current_column = previous.clamp_min(0)
            if cpu and not augmenting.any():
                break
        failed |= augmenting
    return column_for_row, failed


def _assignment_single_cpu(costs: Tensor, integer_costs: bool, sentinel: Tensor) -> tuple[Tensor, Tensor]:
    """Accumulate shortest-path distances and update duals once per augmentation."""
    _, size, width = costs.shape
    matrix = costs[0]
    u, v = costs.new_zeros(size), costs.new_zeros(width)
    row_for_column = torch.full((width,), -1, dtype=torch.long, device=costs.device)
    column_for_row = row_for_column.new_full((size,), -1)
    predecessor = torch.empty_like(row_for_column)
    visited_columns = torch.zeros(width, dtype=torch.bool, device=costs.device)
    distance = costs.new_empty(width)
    failed = torch.zeros(1, dtype=torch.bool, device=costs.device)
    for root in range(size):
        visited_columns.zero_()
        distance.copy_(sentinel)
        current_row = root
        current_column = 0
        minimum = costs.new_zeros(())
        visited_matched = False
        while True:
            reduced = minimum + matrix[current_row] - u[current_row] - v
            improved = ~visited_columns & (reduced < distance)
            distance.copy_(torch.where(improved, reduced, distance))
            predecessor.copy_(torch.where(improved, current_row, predecessor))
            minimum, next_column = torch.where(visited_columns, sentinel, distance).min(dim=0)
            finite = minimum != sentinel if integer_costs else torch.isfinite(minimum)
            if not finite:
                failed.fill_(True)
                return column_for_row.unsqueeze(0), failed
            current_column = int(next_column)
            current_row = int(row_for_column[current_column])
            if current_row < 0:
                break
            visited_columns[current_column] = True
            visited_matched = True
        # Previously matched visited rows are reached via their matched
        # columns. Only these rows and the root need a potential update.
        u[root] += minimum
        if not visited_matched:
            # A first-step free column has distance minimum, so its column
            # potential change is zero and the augmenting path has one edge.
            row_for_column[current_column] = root
            column_for_row[root] = current_column
            continue
        matched_rows = row_for_column[visited_columns]
        matched_distances = distance[visited_columns]
        u[matched_rows] += minimum - matched_distances
        v[visited_columns] -= minimum - matched_distances
        while True:
            current_row = int(predecessor[current_column])
            previous = int(column_for_row[current_row])
            row_for_column[current_column] = current_row
            column_for_row[current_row] = current_column
            if current_row == root:
                break
            current_column = previous
    return column_for_row.unsqueeze(0), failed


# TorchScript is unsupported on Python 3.14+. The same tensor routines run
# eagerly there, keeping the package's advertised Python support intact.
if sys.version_info < (3, 14):
    _assignment_single_cpu = torch.jit.script(_assignment_single_cpu)
    _assignment_core = torch.jit.script(_assignment_core)
