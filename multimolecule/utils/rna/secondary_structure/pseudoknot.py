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


from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from .notations import _greedy_pseudoknot_tiers, _minimal_pseudoknot_tiers

_CROSSING_N2_THRESHOLD = 2048


def split_pseudoknot_pairs(
    pairs: Tensor | np.ndarray | List,
) -> Tuple[Tensor | np.ndarray | List, Tensor | np.ndarray | List]:
    """
    Split pairs into primary (paren-tier) pairs and pseudoknot-tier pairs.

    Args:
        pairs: Tensor/ndarray/list of shape (n, 2) with 0-based indices.

    Returns:
        (primary_pairs, pseudoknot_pairs) with the same backend as input.
        Tiers follow the minimal-coloring assignment used by pairs_to_dot_bracket.
    """
    if isinstance(pairs, Tensor):
        return _torch_split_pseudoknot_pairs(pairs)
    if isinstance(pairs, np.ndarray):
        return _numpy_split_pseudoknot_pairs(pairs)
    if isinstance(pairs, list):
        primary, pseudoknot = _numpy_split_pseudoknot_pairs(np.asarray(pairs, dtype=int))
        return primary.tolist(), pseudoknot.tolist()
    raise ValueError("pairs must be an array-like with shape (n, 2)")


def primary_pairs(pairs: Tensor | np.ndarray | List) -> Tensor | np.ndarray | List:
    """
    Return primary (paren-tier) pairs inferred from dot-bracket tiering.
    """
    primary, _ = split_pseudoknot_pairs(pairs)
    return primary


def pseudoknot_pairs(pairs: Tensor | np.ndarray | List) -> Tensor | np.ndarray | List:
    """
    Return pseudoknot-tier pairs (non-paren tiers).

    Tiers follow the minimal-coloring assignment used by pairs_to_dot_bracket.
    """
    _, pseudoknot = split_pseudoknot_pairs(pairs)
    return pseudoknot


def crossing_pairs(pairs: Tensor | np.ndarray | List) -> Tensor | np.ndarray | List:
    """
    Return pairs that participate in any crossing (pseudoknot events).
    """
    if isinstance(pairs, Tensor):
        norm = _torch_normalize_pairs(pairs)
        if norm.numel() == 0:
            return norm.view(0, 2)
        return norm[_torch_crossing_mask(norm)]
    if isinstance(pairs, list):
        return _numpy_crossing_pairs(np.asarray(pairs, dtype=int)).tolist()
    if isinstance(pairs, np.ndarray):
        return _numpy_crossing_pairs(pairs)
    raise ValueError("pairs must be an array-like with shape (n, 2)")


def pseudoknot_tiers(pairs: Tensor | np.ndarray | List, unsafe: bool = False) -> List:
    """
    Return dot-bracket tiers as non-crossing groups of pairs.

    Args:
        pairs: Tensor/ndarray/list of shape (n, 2) with 0-based indices.
        unsafe: Use greedy tiering for speed instead of minimal coloring.
    """
    if isinstance(pairs, Tensor):
        norm = _torch_normalize_pairs(pairs)
        if norm.numel() == 0:
            return []
        tiers = _tiers_from_numpy(norm.detach().cpu().numpy(), unsafe=unsafe)
        return [torch.from_numpy(tier).to(device=norm.device) for tier in tiers]
    if isinstance(pairs, np.ndarray):
        norm = _numpy_normalize_pairs(pairs)
        return _tiers_from_numpy(norm, unsafe=unsafe)
    if isinstance(pairs, list):
        norm = _numpy_normalize_pairs(np.asarray(pairs, dtype=int))
        tiers = _tiers_from_numpy(norm, unsafe=unsafe)
        return [tier.tolist() for tier in tiers]
    raise ValueError("pairs must be an array-like with shape (n, 2)")


def pseudoknot_nucleotides(pairs: Tensor | np.ndarray | List) -> Tensor | np.ndarray | List:
    """
    Return nucleotide indices involved in any pseudoknot-tier pair.
    """
    if isinstance(pairs, Tensor):
        pkp = _torch_pseudoknot_pairs(pairs)
        if pkp.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=pairs.device)
        return torch.unique(pkp.view(-1)).to(torch.long)
    if isinstance(pairs, list):
        pkp = _numpy_pseudoknot_pairs(np.asarray(pairs, dtype=int))
        return np.unique(pkp.reshape(-1)).tolist() if pkp.size else []
    if isinstance(pairs, np.ndarray):
        pkp = _numpy_pseudoknot_pairs(pairs)
        if pkp.size == 0:
            return np.empty((0,), dtype=int)
        return np.unique(pkp.reshape(-1))
    raise TypeError("pairs must be a numpy.ndarray with shape (n, 2)")


def _tiers_from_numpy(norm: np.ndarray, unsafe: bool = False) -> List[np.ndarray]:
    if norm.size == 0:
        return []
    tiers = _greedy_pseudoknot_tiers(norm) if unsafe else _minimal_pseudoknot_tiers(norm)
    out: List[np.ndarray] = []
    for tier in tiers:
        if tier:
            out.append(np.array(tier, dtype=int))
        else:
            out.append(np.empty((0, 2), dtype=int))
    return out


def _torch_split_pseudoknot_pairs(pairs: Tensor) -> Tuple[Tensor, Tensor]:
    norm = _torch_normalize_pairs(pairs)
    if norm.numel() == 0:
        empty = norm.view(0, 2)
        return empty, empty
    crossing_mask = _torch_crossing_mask(norm)
    if not bool(crossing_mask.any().item()):
        return norm, norm.new_empty((0, 2))
    primary_np, pseudoknot_np = _split_pseudoknot_tiers(norm.detach().cpu().numpy())
    primary = torch.from_numpy(primary_np).to(device=norm.device)
    pseudoknot = torch.from_numpy(pseudoknot_np).to(device=norm.device)
    return primary, pseudoknot


def _numpy_split_pseudoknot_pairs(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm = _numpy_normalize_pairs(pairs)
    if norm.size == 0:
        empty = norm.reshape(0, 2)
        return empty, empty
    return _split_pseudoknot_tiers(norm)


def _split_pseudoknot_tiers(norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if norm.size == 0:
        empty = norm.reshape(0, 2)
        return empty, empty
    empty = norm[:0]
    if norm.shape[0] < 2:
        return norm, empty
    if not _numpy_crossing_mask(norm).any():
        return norm, empty
    tiers = _minimal_pseudoknot_tiers(norm)
    if len(tiers) <= 1:
        return norm, empty
    primary = np.array(tiers[0], dtype=int) if tiers[0] else empty
    pseudoknot_list = [pair for tier in tiers[1:] for pair in tier]
    if pseudoknot_list:
        pseudoknot = np.array(pseudoknot_list, dtype=int)
    else:
        pseudoknot = empty
    return primary, pseudoknot


def _torch_pseudoknot_pairs(pairs: Tensor) -> Tensor:
    norm = _torch_normalize_pairs(pairs)
    if norm.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    crossing_mask = _torch_crossing_mask(norm)
    if not bool(crossing_mask.any().item()):
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    _, pseudoknot_np = _split_pseudoknot_tiers(norm.detach().cpu().numpy())
    return torch.from_numpy(pseudoknot_np).to(device=norm.device, dtype=torch.long)


def _numpy_pseudoknot_pairs(pairs: np.ndarray) -> np.ndarray:
    norm = _numpy_normalize_pairs(pairs)
    if norm.size == 0:
        return np.empty((0, 2), dtype=int)
    _, pseudoknot = _split_pseudoknot_tiers(norm)
    if pseudoknot.size == 0:
        return pseudoknot.reshape(0, 2)
    return pseudoknot.astype(int, copy=False)


def _numpy_crossing_pairs(pairs: np.ndarray) -> np.ndarray:
    norm = _numpy_normalize_pairs(pairs)
    if norm.size == 0:
        return norm.reshape(0, 2)
    mask = _numpy_crossing_mask(norm)
    return norm[mask]


def _torch_normalize_pairs(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0:
        return pairs.view(0, 2).to(dtype=torch.long)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise TypeError("pairs must have shape (n, 2)")
    low = torch.minimum(pairs[:, 0], pairs[:, 1]).to(torch.long)
    high = torch.maximum(pairs[:, 0], pairs[:, 1]).to(torch.long)
    norm = torch.stack([low, high], dim=1)
    if norm.numel() == 0:
        return norm
    if norm.shape[0] > 1:
        ii = norm[:, 0]
        jj = norm[:, 1]
        ii_prev = ii[:-1]
        ii_next = ii[1:]
        jj_prev = jj[:-1]
        jj_next = jj[1:]
        is_sorted = torch.all((ii_next > ii_prev) | ((ii_next == ii_prev) & (jj_next >= jj_prev))).item()
        if is_sorted:
            has_dups = torch.any((ii_next == ii_prev) & (jj_next == jj_prev)).item()
            if not has_dups:
                return norm
            return torch.unique_consecutive(norm, dim=0)
    norm = torch.unique(norm, dim=0)
    if norm.numel() == 0:
        return norm
    key = norm[:, 0] * (int(norm[:, 1].max().item()) + 1 if norm.numel() else 1) + norm[:, 1]
    order = torch.argsort(key)
    return norm[order]


def _numpy_normalize_pairs(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise TypeError("pairs must be a numpy.ndarray with shape (n, 2)")
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    norm = np.column_stack((low, high)).astype(int, copy=False)
    if norm.size == 0:
        return norm
    norm = np.unique(norm, axis=0)
    if norm.size == 0:
        return norm
    ord_idx = np.lexsort((norm[:, 1], norm[:, 0]))
    return norm[ord_idx]


def _torch_crossing_mask(norm_pairs: Tensor) -> Tensor:
    if norm_pairs.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=norm_pairs.device)
    n = norm_pairs.shape[0]
    if n < 2:
        return torch.zeros((n,), dtype=torch.bool, device=norm_pairs.device)
    if norm_pairs.device.type != "cpu" and n <= _CROSSING_N2_THRESHOLD:
        ii = norm_pairs[:, 0]
        jj = norm_pairs[:, 1]
        tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=norm_pairs.device), diagonal=1)
        crosses = (
            tri
            & (ii.unsqueeze(1) < ii.unsqueeze(0))  # noqa: W503
            & (ii.unsqueeze(0) < jj.unsqueeze(1))  # noqa: W503
            & (jj.unsqueeze(1) < jj.unsqueeze(0))  # noqa: W503
        )
        return crosses.any(dim=1) | crosses.any(dim=0)
    return _torch_crossing_mask_large(norm_pairs)


def _torch_crossing_mask_large(norm_pairs: Tensor) -> Tensor:
    if norm_pairs.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=norm_pairs.device)
    n = norm_pairs.shape[0]
    if n < 2:
        return torch.zeros((n,), dtype=torch.bool, device=norm_pairs.device)
    mask_np = _numpy_crossing_mask(norm_pairs.detach().cpu().numpy())
    return torch.from_numpy(mask_np).to(device=norm_pairs.device)


def _numpy_crossing_mask(norm_pairs: np.ndarray) -> np.ndarray:
    if norm_pairs.size == 0:
        return np.zeros((0,), dtype=bool)
    pairs = np.asarray(norm_pairs, dtype=int)
    n = pairs.shape[0]
    if n < 2:
        return np.zeros((n,), dtype=bool)
    ii = pairs[:, 0]
    jj = pairs[:, 1]

    is_sorted = True
    if n > 1:
        ii_prev = ii[:-1]
        ii_next = ii[1:]
        if np.any(ii_prev > ii_next):
            is_sorted = False
        elif np.any(ii_prev == ii_next):
            is_sorted = np.all((ii_prev < ii_next) | (jj[:-1] <= jj[1:]))
    if not is_sorted:
        order = np.lexsort((jj, ii))
        pairs = pairs[order]
        ii = pairs[:, 0]
        jj = pairs[:, 1]
    else:
        order = None

    if n <= _CROSSING_N2_THRESHOLD:
        tri = np.triu(np.ones((n, n), dtype=bool), k=1)
        crosses = tri & (ii[:, None] < ii[None, :]) & (ii[None, :] < jj[:, None]) & (jj[:, None] < jj[None, :])
        mask = crosses.any(axis=1) | crosses.any(axis=0)
    else:
        max_pos = int(max(ii.max(initial=-1), jj.max(initial=-1)))
        length = max_pos + 1
        start_to_end = np.full(length, -1, dtype=np.int64)
        np.maximum.at(start_to_end, ii, jj)

        size = 1
        while size < length:
            size <<= 1
        tree = np.full(2 * size, -1, dtype=np.int64)
        tree[size : size + length] = start_to_end
        for idx in range(size - 1, 0, -1):
            left = tree[idx * 2]
            right = tree[idx * 2 + 1]
            tree[idx] = left if left >= right else right

        def range_max(l: int, r: int) -> int:  # noqa: E741
            if l > r:
                return -1
            l += size
            r += size
            res = -1
            while l <= r:
                if l & 1:
                    if tree[l] > res:
                        res = tree[l]
                    l += 1
                if not (r & 1):
                    if tree[r] > res:
                        res = tree[r]
                    r -= 1
                l >>= 1
                r >>= 1
            return res

        mask_a = np.zeros((n,), dtype=bool)
        for idx in range(n):
            l = int(ii[idx]) + 1  # noqa: E741
            r = int(jj[idx]) - 1
            if l <= r and range_max(l, r) > jj[idx]:
                mask_a[idx] = True

        bit_size = length + 1
        bit = np.zeros(bit_size + 1, dtype=np.int32)

        def bit_add(pos: int) -> None:
            i = pos + 1
            while i <= bit_size:
                bit[i] += 1
                i += i & -i

        def bit_sum(pos: int) -> int:
            if pos < 0:
                return 0
            i = pos + 1
            total = 0
            while i > 0:
                total += bit[i]
                i -= i & -i
            return total

        mask_b = np.zeros((n,), dtype=bool)
        idx = 0
        while idx < n:
            start = int(ii[idx])
            end_idx = idx + 1
            while end_idx < n and ii[end_idx] == start:
                end_idx += 1
            for pos in range(idx, end_idx):
                l = start + 1  # noqa: E741
                r = int(jj[pos]) - 1
                if l <= r and (bit_sum(r) - bit_sum(l - 1)) > 0:
                    mask_b[pos] = True
            for pos in range(idx, end_idx):
                bit_add(int(jj[pos]))
            idx = end_idx

        mask = mask_a | mask_b

    if order is not None:
        mask_unsorted = np.empty_like(mask)
        mask_unsorted[order] = mask
        return mask_unsorted
    return mask


def _torch_range_max(start: Tensor, l: Tensor, r: Tensor) -> Tensor:  # noqa: E741
    if l.numel() == 0:
        return l.to(dtype=torch.long)
    out = torch.full_like(l, -1, dtype=torch.long)
    if start.numel() == 0:
        return out
    for idx in range(l.numel()):
        li = int(l[idx].item())
        ri = int(r[idx].item())
        if li <= ri:
            out[idx] = torch.max(start[li : ri + 1]).to(torch.long)
    return out


def _torch_crossing_mask_range_max(ii: Tensor, jj: Tensor, length: int) -> Tensor:
    if ii.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=ii.device)
    pairs = torch.stack((ii, jj), dim=1)
    return _torch_crossing_mask_large(pairs)
