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

"""Graph aggregation operations.

Pure-PyTorch replacements for torch_scatter:
- scatter: general scatter aggregation (sum, mean, min, max)
- scatter_mean: scatter with mean reduction
"""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("scatter does not support scalar src tensors")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dim must be in the range [{-ndim}, {ndim - 1}], but got {dim}")
    return dim


def _broadcast_index(index: Tensor, src: Tensor, dim: int) -> Tensor:
    if index.dim() == 1:
        if index.numel() != src.size(dim):
            raise ValueError(f"1D index length must match src.size(dim), got {index.numel()} and {src.size(dim)}")
        shape = [1] * src.dim()
        shape[dim] = -1
        return index.view(shape).expand_as(src)

    if index.dim() != src.dim():
        raise ValueError(
            f"index must be 1D or have the same number of dimensions as src, got {index.dim()} and {src.dim()}"
        )
    try:
        return index.expand_as(src)
    except RuntimeError as error:
        raise ValueError(
            f"index with shape {tuple(index.shape)} cannot be broadcast to src shape {tuple(src.shape)}"
        ) from error


def _output_size(index: Tensor, dim_size: int | None) -> int:
    if torch.compiler.is_compiling() and dim_size is not None:
        if dim_size < 0:
            raise ValueError(f"dim_size must be non-negative, but got {dim_size}")
        if index.numel():
            # Keep validation on the device: Python boolean conversion would
            # synchronize CUDA and break fullgraph compilation.
            torch._assert_async(
                ((index >= 0) & (index < dim_size)).all(),
                "index must contain values in [0, dim_size)",
            )
        return dim_size

    if index.numel():
        minimum, maximum = torch.aminmax(index)
        min_index, max_index = torch.stack((minimum, maximum)).tolist()
        if min_index < 0:
            raise ValueError("index must contain non-negative values")
    else:
        max_index = -1
    if dim_size is None:
        dim_size = max_index + 1
    if dim_size < 0:
        raise ValueError(f"dim_size must be non-negative, but got {dim_size}")
    if max_index >= dim_size:
        raise ValueError("index contains values greater than or equal to dim_size")
    return dim_size


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> Tensor:
    """Scatter operation: aggregate src values by index.

    Args:
        src: Source tensor.
        index: Index tensor, same size as src along dim.
        dim: Dimension along which to scatter.
        dim_size: Size of the output along dim.
        reduce: Aggregation method ('sum', 'mean', 'min', 'max').

    Returns:
        Aggregated tensor.

    Mean reduction accumulates float16/bfloat16 sources in float32 and integer
    or bool sources in float64, with int64 counts. Other floating and complex
    sources retain their accumulation dtype. Floating and complex means retain
    the source dtype; integer and bool means return float32. Integer conversion
    to float64 can round values beyond ``2**53``, so integer means are not
    guaranteed to equal the exact real-valued average.

    With an explicit ``dim_size``, supports fullgraph compilation. Eager index
    validation raises ``ValueError``; compiled validation uses a device assertion
    and may raise ``RuntimeError`` instead. CUDA assertions are asynchronous.
    """
    dim = _normalize_dim(dim, src.dim())
    if index.dtype == torch.bool or torch.is_floating_point(index) or torch.is_complex(index):
        raise TypeError("index must be an integer tensor")
    index = index.to(torch.long)
    idx = _broadcast_index(index, src, dim)
    dim_size = _output_size(index, dim_size)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    accumulator_dtype = src.dtype
    if reduce == "mean":
        if src.dtype in (torch.float16, torch.bfloat16):
            accumulator_dtype = torch.float32
        elif not src.is_floating_point() and not src.is_complex():
            accumulator_dtype = torch.float64
    out = torch.zeros(*out_shape, dtype=accumulator_dtype, device=src.device)

    if reduce == "sum":
        out.scatter_add_(dim, idx, src)
    elif reduce == "mean":
        out.scatter_add_(dim, idx, src.to(accumulator_dtype))
        count_dtype = torch.int64
        if index.dim() == 1:
            count = torch.zeros(dim_size, dtype=count_dtype, device=src.device)
            count.scatter_add_(0, index, torch.ones_like(index, dtype=count_dtype))
            shape = [1] * src.dim()
            shape[dim] = dim_size
            out = out / count.clamp_min(1).view(shape)
        else:
            count = torch.zeros(*out_shape, dtype=count_dtype, device=src.device)
            ones = torch.ones((), dtype=count_dtype, device=src.device).expand_as(src)
            count.scatter_add_(dim, idx, ones)
            out = out / count.clamp_min(1)
        result_dtype = src.dtype if src.is_floating_point() or src.is_complex() else torch.float32
        out = out.to(result_dtype)
    elif reduce == "min":
        out.scatter_reduce_(dim, idx, src, reduce="amin", include_self=False)
    elif reduce == "max":
        out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")

    return out


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Average values by index, using the accumulation and output dtypes of :func:`scatter`."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
