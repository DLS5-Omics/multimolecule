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

"""Diffusion process utilities for DiffDock.

Provides noise schedules, timestep embeddings, conformer modification under diffusion,
and helper functions for the forward/reverse diffusion process.

Rewritten to operate on dict-based data instead of PyG HeteroData.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta
from torch import Tensor, nn

from .geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch, rigid_transform_Kabsch_3D_torch_batch
from .torsion import modify_conformer_torsion_angles, modify_conformer_torsion_angles_batch


def sigmoid(t: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.e ** (-t))


def sigmoid_schedule(t: float | np.ndarray, k: float = 10, m: float = 0.5) -> float | np.ndarray:
    s = lambda t: sigmoid(k * (t - m))
    return (s(t) - s(0)) / (s(1) - s(0))


def t_to_sigma(t_tr: Tensor, t_rot: Tensor, t_tor: Tensor, args: Any) -> tuple[Tensor, Tensor, Tensor]:
    """Convert diffusion timesteps to noise levels (sigma) for TR/ROT/TOR."""
    tr_sigma = args.tr_sigma_min ** (1 - t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1 - t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1 - t_tor) * args.tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(
    data: dict[str, Any],
    tr_update: Tensor,
    rot_update: Tensor,
    torsion_updates: Tensor | None,
    pivot: Tensor | None = None,
) -> dict[str, Any]:
    """Apply translation, rotation, and torsion updates to the ligand conformer.

    Operates on dict-based data structure (no PyG HeteroData).
    """
    lig_center = torch.mean(data["ligand_pos"], dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (data["ligand_pos"] - lig_center) @ rot_mat.T + tr_update + lig_center

    if torsion_updates is not None:
        edge_index = data["ligand_edge_index"].T[data["ligand_edge_mask"]]
        mask_rotate = data["ligand_mask_rotate"]
        if isinstance(mask_rotate, list):
            mask_rotate = mask_rotate[0]

        flexible_new_pos = modify_conformer_torsion_angles(
            rigid_new_pos, edge_index, mask_rotate, torsion_updates
        ).to(rigid_new_pos.device)

        if pivot is None:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        else:
            R1, t1 = rigid_transform_Kabsch_3D_torch(pivot.T, rigid_new_pos.T)
            R2, t2 = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, pivot.T)
            aligned_flexible_pos = (flexible_new_pos @ R2.T + t2.T) @ R1.T + t1.T

        data["ligand_pos"] = aligned_flexible_pos
    else:
        data["ligand_pos"] = rigid_new_pos

    return data


def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int, max_positions: int = 10000) -> Tensor:
    """Sinusoidal positional embedding for diffusion timesteps."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian random Fourier features for noise level embedding."""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_timestep_embedding(
    embedding_type: str, embedding_dim: int, embedding_scale: int = 10000
) -> Callable:
    """Create a timestep embedding function."""
    if embedding_type == "sinusoidal":
        return lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim)
    elif embedding_type == "fourier":
        return GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    raise NotImplementedError(f"Unknown embedding type: {embedding_type}")


def get_t_schedule(
    sigma_schedule: str,
    inference_steps: int,
    inf_sched_alpha: float = 1,
    inf_sched_beta: float = 1,
    t_max: float = 1,
) -> np.ndarray:
    """Generate a timestep schedule for inference."""
    if sigma_schedule == "expbeta":
        lin_max = beta.cdf(t_max, a=inf_sched_alpha, b=inf_sched_beta)
        c = np.linspace(lin_max, 0, inference_steps + 1)[:-1]
        return beta.ppf(c, a=inf_sched_alpha, b=inf_sched_beta)
    raise ValueError(f"Unknown sigma schedule: {sigma_schedule}")


def set_time(
    data: dict[str, Any],
    t_tr: float,
    t_rot: float,
    t_tor: float,
    batchsize: int,
    device: torch.device,
) -> None:
    """Set diffusion timestep tensors on the data dict.

    Operates on dict-based data structure (no PyG HeteroData).
    """
    n_lig = len(data["ligand_x"]) if "ligand_x" in data else len(data["ligand_pos"])
    n_rec = len(data["receptor_x"]) if "receptor_x" in data else len(data["receptor_pos"])

    data["ligand_node_t"] = {
        "tr": t_tr * torch.ones(n_lig, device=device),
        "rot": t_rot * torch.ones(n_lig, device=device),
        "tor": t_tor * torch.ones(n_lig, device=device),
    }
    data["receptor_node_t"] = {
        "tr": t_tr * torch.ones(n_rec, device=device),
        "rot": t_rot * torch.ones(n_rec, device=device),
        "tor": t_tor * torch.ones(n_rec, device=device),
    }
    data["complex_t"] = {
        "tr": t_tr * torch.ones(batchsize, device=device),
        "rot": t_rot * torch.ones(batchsize, device=device),
        "tor": t_tor * torch.ones(batchsize, device=device),
    }
    if "atom_x" in data or "atom_pos" in data:
        n_atom = len(data.get("atom_x", data.get("atom_pos")))
        data["atom_node_t"] = {
            "tr": t_tr * torch.ones(n_atom, device=device),
            "rot": t_rot * torch.ones(n_atom, device=device),
            "tor": t_tor * torch.ones(n_atom, device=device),
        }
