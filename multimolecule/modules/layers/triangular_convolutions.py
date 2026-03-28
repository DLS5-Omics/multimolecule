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

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_square_spatial(tensor: Tensor, op_name: str, tensor_name: str) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{op_name} expected a 4D {tensor_name} tensor, got shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(
            f"{op_name} requires square {tensor_name} spatial dimensions, got {tuple(tensor.shape[-2:])}."
        )


def _conv2d_output_shape(
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    out_h = ((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
    out_w = ((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
    return out_h, out_w


def _conv_transpose2d_output_shape(
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    output_padding: Tuple[int, int],
) -> Tuple[int, int]:
    out_h = (height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    out_w = (width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    return out_h, out_w


def _scatter_upper_triangle(upper: Tensor, row_idx: Tensor, col_idx: Tensor, size: int) -> Tensor:
    output = upper.new_zeros((upper.shape[0], upper.shape[1], size, size))
    output[:, :, row_idx, col_idx] = upper
    non_diag = row_idx != col_idx
    if bool(non_diag.any()):
        output[:, :, col_idx[non_diag], row_idx[non_diag]] = upper[:, :, non_diag]
    return output


def _upper_triangle_indices(size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    return torch.triu_indices(size, size, device=device)


class TriangularConv2d(nn.Conv2d):
    r"""
    Convolution over only the non-redundant upper triangle of a square contact map.

    The reference implementation computes only the upper-triangular output locations via
    ``torch.nn.functional.unfold`` and mirrors them to the lower triangle, producing a
    symmetric dense output without evaluating lower-triangular output positions.
    """

    def _reference_forward(self, input: Tensor) -> Tensor:
        _require_square_spatial(input, "TriangularConv2d", "input")
        out_h, out_w = _conv2d_output_shape(
            input.shape[-2], input.shape[-1], self.kernel_size, self.stride, self.padding, self.dilation
        )
        if out_h != out_w:
            raise ValueError(f"TriangularConv2d requires square output spatial dimensions, got {(out_h, out_w)}.")

        patches = F.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        row_idx, col_idx = _upper_triangle_indices(out_h, input.device)
        flat_idx = row_idx * out_w + col_idx
        upper_patches = patches.index_select(2, flat_idx)
        upper_patches = upper_patches.reshape(input.shape[0], self.groups, -1, flat_idx.shape[0])
        weight = self.weight.reshape(self.groups, self.out_channels // self.groups, -1)
        upper = torch.einsum("goc,ngcu->ngou", weight, upper_patches)
        upper = upper.reshape(input.shape[0], self.out_channels, flat_idx.shape[0])
        if self.bias is not None:
            upper = upper + self.bias.view(1, -1, 1)
        return _scatter_upper_triangle(upper, row_idx, col_idx, out_h)

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            raise NotImplementedError("TriangularConv2d only supports zero padding.")
        if isinstance(self.padding, str):
            raise NotImplementedError("TriangularConv2d does not support string padding modes.")
        return self._reference_forward(input)


class TriangularConvTranspose2d(nn.ConvTranspose2d):
    r"""
    Transposed convolution with triangular output semantics.

    This reference implementation computes only upper-triangular output locations and
    mirrors them to the lower triangle, without materializing dense output activations.
    """

    def _reference_forward(self, input: Tensor, output_padding: Tuple[int, int]) -> Tensor:
        _require_square_spatial(input, "TriangularConvTranspose2d", "input")
        out_h, out_w = _conv_transpose2d_output_shape(
            input.shape[-2],
            input.shape[-1],
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            output_padding,
        )
        if out_h != out_w:
            raise ValueError(
                f"TriangularConvTranspose2d requires square output spatial dimensions, got {(out_h, out_w)}."
            )

        row_idx, col_idx = _upper_triangle_indices(out_h, input.device)
        upper = input.new_zeros((input.shape[0], self.out_channels, row_idx.shape[0]))

        in_channels_per_group = self.in_channels // self.groups
        out_channels_per_group = self.out_channels // self.groups

        for kernel_row in range(self.kernel_size[0]):
            row_numerator = row_idx + self.padding[0] - kernel_row * self.dilation[0]
            valid_rows = (row_numerator >= 0) & (row_numerator % self.stride[0] == 0)
            input_rows = torch.div(row_numerator, self.stride[0], rounding_mode="floor")
            valid_rows = valid_rows & (input_rows < input.shape[-2])

            for kernel_col in range(self.kernel_size[1]):
                col_numerator = col_idx + self.padding[1] - kernel_col * self.dilation[1]
                valid_cols = (col_numerator >= 0) & (col_numerator % self.stride[1] == 0)
                input_cols = torch.div(col_numerator, self.stride[1], rounding_mode="floor")
                valid_cols = valid_cols & (input_cols < input.shape[-1])

                valid = valid_rows & valid_cols
                if not bool(valid.any()):
                    continue

                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                sampled = input[:, :, input_rows[valid_idx], input_cols[valid_idx]]
                sampled = sampled.reshape(input.shape[0], self.groups, in_channels_per_group, valid_idx.shape[0])
                weight = self.weight[:, :, kernel_row, kernel_col].reshape(
                    self.groups, in_channels_per_group, out_channels_per_group
                )
                contrib = torch.einsum("ngcv,gco->ngov", sampled, weight)
                contrib = contrib.reshape(input.shape[0], self.out_channels, valid_idx.shape[0])
                upper.index_add_(2, valid_idx, contrib)

        if self.bias is not None:
            upper = upper + self.bias.view(1, -1, 1)
        return _scatter_upper_triangle(upper, row_idx, col_idx, out_h)

    def forward(self, input: Tensor, output_size: Tuple[int, int] | None = None) -> Tensor:
        if isinstance(self.padding, str):
            raise NotImplementedError("TriangularConvTranspose2d does not support string padding modes.")

        output_padding = tuple(
            self._output_padding(  # type: ignore[misc]
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                2,
                self.dilation,
            )
        )
        return self._reference_forward(input, output_padding)


__all__ = [
    "TriangularConv2d",
    "TriangularConvTranspose2d",
]
