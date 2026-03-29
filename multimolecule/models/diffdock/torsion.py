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

"""Torsion angle utilities for DiffDock.

Handles torsion angle modification of molecular conformers and identification
of rotatable bonds. Rewritten without PyG dependencies (replaces to_networkx + PyG Data).
"""

from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from .geometry import axis_angle_to_matrix


def get_transformation_mask(
    edge_index: Tensor,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify rotatable bonds and build rotation masks.

    Replaces the original get_transformation_mask that used PyG's to_networkx.

    Args:
        edge_index: Bond edge index of shape (2, E) for the ligand.
        num_nodes: Number of ligand atoms.

    Returns:
        mask_edges: Boolean array (E,) indicating which edges are rotatable.
        mask_rotate: Boolean array (num_rotatable, num_nodes) indicating which
            atoms rotate when each rotatable bond is twisted.
    """
    # Build networkx graph from edge_index directly
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.T.cpu().numpy()
    for e in edges:
        G.add_edge(int(e[0]), int(e[1]))

    to_rotate = []
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i + 1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(int(edges[i, 0]), int(edges[i, 1]))
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), num_nodes), dtype=bool)
    idx = 0
    for i in range(min(edges.shape[0], len(G.edges()))):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def modify_conformer_torsion_angles(
    pos: Tensor | np.ndarray,
    edge_index: np.ndarray | Tensor,
    mask_rotate: np.ndarray | list,
    torsion_updates: np.ndarray | Tensor,
    as_numpy: bool = False,
) -> Tensor | np.ndarray:
    """Apply torsion angle updates to a molecular conformer.

    Args:
        pos: Atom positions, shape (N, 3).
        edge_index: Rotatable bond edges, shape (num_rotatable, 2).
        mask_rotate: Boolean mask, shape (num_rotatable, N).
        torsion_updates: Angle updates in radians, shape (num_rotatable,).
        as_numpy: Return numpy array instead of tensor.

    Returns:
        Updated atom positions.
    """
    pos = copy.deepcopy(pos)
    if not isinstance(pos, np.ndarray):
        pos = pos.cpu().numpy()

    if isinstance(mask_rotate, list):
        mask_rotate = mask_rotate[0]
    if isinstance(edge_index, Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(torsion_updates, Tensor):
        torsion_updates = torsion_updates.cpu().numpy()

    for idx_edge, e in enumerate(edge_index):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = int(e[0]), int(e[1])

        rot_vec = pos[u] - pos[v]
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy:
        pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def modify_conformer_torsion_angles_batch(
    pos: Tensor,
    edge_index: Tensor,
    mask_rotate: np.ndarray,
    torsion_updates: Tensor,
) -> Tensor:
    """Batched torsion angle modification.

    Args:
        pos: Atom positions, shape (B, N, 3).
        edge_index: Rotatable bond edges, shape (num_rotatable, 2).
        mask_rotate: Boolean mask, shape (num_rotatable, N).
        torsion_updates: Angle updates, shape (B, num_rotatable).

    Returns:
        Updated positions, shape (B, N, 3).
    """
    pos = pos + 0  # clone
    for idx_edge, e in enumerate(edge_index):
        u, v = e[0], e[1]

        rot_vec = pos[:, u] - pos[:, v]
        rot_mat = axis_angle_to_matrix(
            rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True)
            * torsion_updates[:, idx_edge : idx_edge + 1]
        )

        pos[:, mask_rotate[idx_edge]] = (
            torch.bmm(
                pos[:, mask_rotate[idx_edge]] - pos[:, v : v + 1],
                torch.transpose(rot_mat, 1, 2),
            )
            + pos[:, v : v + 1]
        )

    return pos
