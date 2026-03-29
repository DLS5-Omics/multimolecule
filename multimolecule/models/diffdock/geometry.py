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

"""Rotation and rigid-body geometry utilities for DiffDock.

Pure PyTorch implementations of rotation conversions (axis-angle, quaternion, matrix)
and Kabsch alignment. Adapted from PyTorch3D rotation_conversions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert quaternions (real-first) to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to quaternions (real-first)."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def _sqrt_positive_part(x: Tensor) -> Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to quaternions (real-first)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack([
            1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
        ], dim=-1)
    )
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_axis_angle(quaternions: Tensor) -> Tensor:
    """Convert quaternions (real-first) to axis-angle vectors."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rigid_transform_Kabsch_3D_torch(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Kabsch alignment: find R, t such that B ~ R @ A + t.

    Args:
        A: Source points, shape (3, N).
        B: Target points, shape (3, N).

    Returns:
        R: Rotation matrix (3, 3).
        t: Translation vector (3, 1).
    """
    assert A.shape[1] == B.shape[1]
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ Bm.T
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3
    t = -R @ centroid_A + centroid_B
    return R, t


def rigid_transform_Kabsch_3D_torch_batch(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Batched Kabsch alignment.

    Args:
        A: Source points, shape (B, N, 3).
        B: Target points, shape (B, N, 3).

    Returns:
        R: Rotation matrices (B, 3, 3).
        t: Translation vectors (B, 3, 1).
    """
    assert A.shape == B.shape
    A, B = A.permute(0, 2, 1), B.permute(0, 2, 1)
    centroid_A = torch.mean(A, axis=2, keepdims=True)
    centroid_B = torch.mean(B, axis=2, keepdims=True)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = torch.bmm(Am, Bm.transpose(1, 2))
    U, S, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
    Rm = torch.bmm(Vt.transpose(1, 2) @ SS, U.transpose(1, 2))
    R = torch.where(torch.linalg.det(R)[:, None, None] < 0, Rm, R)
    t = torch.bmm(-R, centroid_A) + centroid_B
    return R, t
