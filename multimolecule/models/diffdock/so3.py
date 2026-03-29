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

"""SO(3) diffusion utilities for rotational score computation.

Precomputes and caches truncated infinite series for:
- Sampling from the IGSO(3) distribution
- Computing the score of the IGSO(3) density
- Expected score norms for training normalization

The cache is saved to disk on first run and loaded on subsequent runs.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

MIN_EPS, MAX_EPS, N_EPS = 0.0005, 4, 2000
X_N = 2000

omegas = np.linspace(0, np.pi, X_N + 1)[1:]

_CACHE_DIR = os.path.dirname(__file__)


def _compose(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Compose two axis-angle rotations."""
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega: np.ndarray, eps: float, L: int = 2000) -> np.ndarray:
    """Truncated infinite series expansion for IGSO(3) density."""
    l_vec = np.arange(L).reshape(-1, 1)
    p = ((2 * l_vec + 1) * np.exp(-l_vec * (l_vec + 1) * eps ** 2 / 2)
         * np.sin(omega * (l_vec + 1 / 2)) / np.sin(omega / 2)).sum(0)
    return p


def _density(expansion: np.ndarray, omega: np.ndarray, marginal: bool = True) -> np.ndarray:
    """IGSO(3) density: marginal over [0, pi] or full over SO(3)."""
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    return expansion / 8 / np.pi ** 2


def _score(exp: np.ndarray, omega: np.ndarray, eps: float, L: int = 2000) -> np.ndarray:
    """Score (gradient of log-density) of IGSO(3)."""
    l_vec = np.arange(L).reshape(-1, 1)
    hi = np.sin((l_vec + 1 / 2) * omega)
    dhi = (l_vec + 1 / 2) * np.cos((l_vec + 1 / 2) * omega)
    lo = np.sin(omega / 2)
    dlo = 1 / 2 * np.cos(omega / 2)
    dSigma = ((2 * l_vec + 1) * np.exp(-l_vec * (l_vec + 1) * eps**2 / 2)
              * (lo * dhi - hi * dlo) / lo ** 2).sum(0)
    return dSigma / exp


def _load_or_compute_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached SO(3) arrays or compute and save them."""
    cache_prefix = os.path.join(_CACHE_DIR, ".so3_cache")
    paths = [f"{cache_prefix}_omegas.npy", f"{cache_prefix}_cdf.npy",
             f"{cache_prefix}_score_norms.npy", f"{cache_prefix}_exp_score_norms.npy"]

    if all(os.path.exists(p) for p in paths):
        return tuple(np.load(p) for p in paths)  # type: ignore[return-value]

    eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    exp_vals = np.asarray([_expansion(omegas_array, eps) for eps in eps_array])
    pdf_vals = np.asarray([_density(exp, omegas_array, marginal=True) for exp in exp_vals])
    cdf_vals = np.asarray([pdf.cumsum() / X_N * np.pi for pdf in pdf_vals])
    score_norms_arr = np.asarray([_score(exp_vals[i], omegas_array, eps_array[i]) for i in range(len(eps_array))])
    exp_score_norms = np.sqrt(
        np.sum(score_norms_arr**2 * pdf_vals, axis=1) / np.sum(pdf_vals, axis=1) / np.pi
    )

    for p, arr in zip(paths, [omegas_array, cdf_vals, score_norms_arr, exp_score_norms]):
        np.save(p, arr)

    return omegas_array, cdf_vals, score_norms_arr, exp_score_norms


_omegas_array, _cdf_vals, _score_norms, _exp_score_norms = _load_or_compute_cache()


def sample(eps: float) -> float:
    """Sample a rotation angle from IGSO(3) with noise level eps."""
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps: float) -> np.ndarray:
    """Sample a random rotation vector from IGSO(3) with noise level eps."""
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps: float, vec: np.ndarray) -> np.ndarray:
    """Compute the score vector for a rotation."""
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps: torch.Tensor) -> torch.Tensor:
    """Expected score norm for given noise levels (used for score normalization)."""
    eps_np = eps.numpy()
    eps_idx = (np.log10(eps_np) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()
