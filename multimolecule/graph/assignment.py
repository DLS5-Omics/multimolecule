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
    potential updates. CUDA search and augmentation loops use shape-derived
    bounds and active masks; validation and the final feasibility check may
    synchronize. CPU searches may exit early. No graph data leaves its device.
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
    size, width = min(rows, columns), max(rows, columns)
    batch_size = prod(batch_shape)
    if size == 0 or batch_size == 0:
        return torch.empty((*batch_shape, size, 2), dtype=torch.long, device=cost_matrix.device)

    costs = cost_matrix.detach().to(torch.int64 if integer_costs else torch.float64)
    costs = costs.reshape(batch_size, rows, columns)
    transposed = rows > columns
    if transposed:
        costs = costs.transpose(1, 2)

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

    # Column zero is the dummy root. p maps columns to 1-based matched rows.
    u = costs.new_zeros((batch_size, size + 1))
    v = costs.new_zeros((batch_size, width + 1))
    p = torch.zeros((batch_size, width + 1), dtype=torch.long, device=costs.device)
    row_ids = torch.arange(size + 1, dtype=torch.long, device=costs.device)
    failed = torch.zeros(batch_size, dtype=torch.bool, device=costs.device)
    cpu = costs.device.type == "cpu"

    for row in range(1, size + 1):
        p[:, 0].copy_(row_ids[row].expand(batch_size))
        current = p.new_zeros(batch_size)
        way = torch.zeros_like(p)
        used = torch.zeros_like(p, dtype=torch.bool)
        minimum = costs.new_full((batch_size, width), sentinel)
        active = ~failed

        # At most row-1 occupied columns precede a free column.
        for _ in range(row):
            current_column = current.unsqueeze(1)
            used.scatter_(1, current_column, used.gather(1, current_column) | active.unsqueeze(1))
            matched_row = p.gather(1, current_column)
            selected_rows = (matched_row - 1).clamp_min(0).unsqueeze(2).expand(-1, -1, width)
            selected_costs = costs.gather(1, selected_rows).squeeze(1)
            reduced = selected_costs - u.gather(1, matched_row) - v[:, 1:]
            available = ~used[:, 1:]
            improved = active.unsqueeze(1) & available & (reduced < minimum)
            minimum = torch.where(improved, reduced, minimum)
            way[:, 1:] = torch.where(improved, current_column, way[:, 1:])
            delta, next_column = torch.where(available, minimum, sentinel).min(dim=1)
            next_column = next_column + 1
            finite = delta != sentinel if integer_costs else torch.isfinite(delta)
            step = active & finite
            failed = failed | (active & ~finite)
            # Mask before arithmetic: an infeasible search must never use inf.
            delta = torch.where(step, delta, 0)
            potential_change = torch.where(used & step.unsqueeze(1), delta.unsqueeze(1), 0)
            # Every nonzero update has a distinct matched-row target. Repeated
            # p=0 indices belong to unused columns and contribute zero.
            u.scatter_add_(1, p, potential_change)
            v = v - potential_change
            minimum = torch.where(
                step.unsqueeze(1) & available & (minimum != sentinel), minimum - delta.unsqueeze(1), minimum
            )
            current = torch.where(step, next_column, current)
            active = step & (p.gather(1, next_column.unsqueeze(1)).squeeze(1) != 0)
            if cpu and not active.any():
                break
        failed = failed | active

        augmenting = ~failed
        for _ in range(row):
            current_column = current.unsqueeze(1)
            previous = way.gather(1, current_column)
            replacement = p.gather(1, previous)
            p.scatter_(
                1, current_column, torch.where(augmenting.unsqueeze(1), replacement, p.gather(1, current_column))
            )
            current = torch.where(augmenting, previous.squeeze(1), current)
            augmenting = augmenting & (current != 0)
            if cpu and not augmenting.any():
                break
        failed = failed | augmenting

    column_for_row = p.new_full((batch_size, size + 1), -1)
    column_indices = torch.arange(width, device=costs.device).expand(batch_size, -1)
    # Unmatched columns collide only at row zero, which is discarded.
    column_for_row.scatter_(1, p[:, 1:], column_indices)
    column_for_row = column_for_row[:, 1:]
    failed = failed | (column_for_row < 0).any(dim=1)
    if not integer_costs:
        selected = costs.gather(2, column_for_row.clamp_min(0).unsqueeze(2)).squeeze(2)
        failed = failed | ~torch.isfinite(selected).all(dim=1)
    if failed.any():
        raise ValueError("cost_matrix is infeasible: no full assignment covers the smaller dimension.")

    row_indices = torch.arange(size, device=costs.device).expand(batch_size, -1)
    if transposed:
        order = column_for_row.argsort(dim=1)
        pairs = torch.stack((column_for_row.gather(1, order), row_indices.gather(1, order)), dim=2)
    else:
        pairs = torch.stack((row_indices, column_for_row), dim=2)
    return pairs.reshape(*batch_shape, size, 2)
