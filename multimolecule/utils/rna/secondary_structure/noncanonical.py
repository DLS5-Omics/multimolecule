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

from typing import Sequence
from warnings import warn

import numpy as np
import torch
from torch import Tensor

from .pairs import (
    Pair,
    Pairs,
    PairsList,
    _numpy_normalize_pairs_low_high,
    _numpy_sort_pairs,
    _torch_normalize_pairs_low_high,
    _torch_sort_pairs,
)

WATSON_CRICK_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G")}
CANONICAL_PAIRS = WATSON_CRICK_PAIRS | WOBBLE_PAIRS
_BASE_CODE_TABLE = np.full(256, -1, dtype=np.int8)
_BASE_CODE_TABLE[ord("A")] = 0
_BASE_CODE_TABLE[ord("C")] = 1
_BASE_CODE_TABLE[ord("G")] = 2
_BASE_CODE_TABLE[ord("U")] = 3
_CANONICAL_TABLE = np.zeros((4, 4), dtype=bool)
for _left, _right in CANONICAL_PAIRS:
    _CANONICAL_TABLE[_BASE_CODE_TABLE[ord(_left)], _BASE_CODE_TABLE[ord(_right)]] = True
_CANONICAL_TABLE_TORCH = torch.from_numpy(_CANONICAL_TABLE)
_CANONICAL_TABLE_TORCH_CACHE = {_CANONICAL_TABLE_TORCH.device: _CANONICAL_TABLE_TORCH}


def noncanonical_pairs(
    pairs: Tensor | np.ndarray | Pairs, sequence: str, unsafe: bool = False
) -> Tensor | np.ndarray | PairsList:
    """
    Return subset of base pairs that are non-canonical (backend-aware).

    Non-ACGU bases are treated as non-canonical.

    Args:
        pairs: Base pairs as a (n, 2) tensor or array.
        sequence: RNA sequence string.
        unsafe: Ignore invalid self-pairs when True; raise when False.

    Returns:
        Non-canonical base pairs using the same backend as input (list inputs return list of tuples).

    Examples:
        Torch input
        >>> import torch
        >>> noncanonical_pairs(torch.tensor([[0, 3]]), "ACGU").tolist()
        []
        >>> noncanonical_pairs(torch.tensor([[0, 3]]), "ACGA").tolist()
        [[0, 3]]
        >>> noncanonical_pairs(torch.tensor([[0, 1]]), "GU").tolist()
        []

        NumPy input
        >>> import numpy as np
        >>> noncanonical_pairs(np.array([[0, 3]]), "ACGU").tolist()
        []
        >>> noncanonical_pairs(np.array([[0, 3]]), "ACGA").tolist()
        [[0, 3]]
        >>> noncanonical_pairs(np.array([[0, 1]]), "GU").tolist()
        []
        >>> noncanonical_pairs(np.array([[0, 1]]), "AX").tolist()
        [[0, 1]]
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_noncanonical_pairs(pairs, sequence, unsafe)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_noncanonical_pairs(pairs, sequence, unsafe)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return list(map(tuple, _numpy_noncanonical_pairs(pairs, sequence, unsafe).tolist()))
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_noncanonical_pairs(pairs: Tensor, sequence: str, unsafe: bool) -> Tensor:
    if pairs.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    mask = _torch_noncanonical_pair_mask(pairs[:, 0], pairs[:, 1], sequence, unsafe)
    if not mask.any():
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    out = _torch_normalize_pairs_low_high(pairs[mask])
    return _torch_sort_pairs(out)


def _numpy_noncanonical_pairs(pairs: np.ndarray, sequence: str, unsafe: bool) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    mask = _numpy_noncanonical_pair_mask(pairs[:, 0], pairs[:, 1], sequence, unsafe)
    if not np.any(mask):
        return np.empty((0, 2), dtype=int)
    out = _numpy_normalize_pairs_low_high(pairs[mask])
    return _numpy_sort_pairs(out)


def noncanonical_pairs_set(
    pairs: Tensor | np.ndarray | Pairs,
    sequence: str,
    unsafe: bool = False,
) -> set[Pair]:
    """
    Return non-canonical base pairs as a set of ``(i, j)`` tuples.

    Args:
        pairs: Base pairs as a tensor, array, or sequence of (i, j) tuples.
        sequence: RNA sequence string.
        unsafe: Ignore invalid self-pairs when True; raise when False.

    Returns:
        A set of (i, j) tuples for non-canonical pairs.

    Examples:
        >>> sorted(noncanonical_pairs_set([(0, 3), (1, 2)], "ACGA"))
        [(0, 3)]
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            return set()
        pairs = pairs.detach().cpu().numpy()
    elif isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            return set()
    elif isinstance(pairs, Sequence):
        if not pairs:
            return set()
        pairs = np.asarray(pairs, dtype=int)
    else:
        raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")

    if pairs.size == 0:
        return set()
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a sequence/array of (i, j) index tuples")

    pairs = pairs.astype(int, copy=False)
    pairs = _numpy_normalize_pairs_low_high(pairs)
    mask = noncanonical_pair_mask(pairs[:, 0], pairs[:, 1], sequence, unsafe)
    if not np.any(mask):
        return set()
    return {(int(pairs[idx, 0]), int(pairs[idx, 1])) for idx in np.flatnonzero(mask)}


def noncanonical_pair_mask(
    pair_i: Tensor | np.ndarray | Sequence[int],
    pair_j: Tensor | np.ndarray | Sequence[int],
    sequence: str,
    unsafe: bool = False,
) -> Tensor | np.ndarray | list[bool]:
    """
    Return a boolean mask indicating which pairs are non-canonical (backend-aware).

    Non-ACGU bases are treated as non-canonical.

    Args:
        pair_i: First nucleotide indices as a tensor or array.
        pair_j: Second nucleotide indices as a tensor or array.
        sequence: RNA sequence string.
        unsafe: Ignore invalid self-pairs when True; raise when False.

    Returns:
        Boolean mask where True indicates a non-canonical pair (same backend as input).

    Examples:
        Torch input
        >>> import torch
        >>> noncanonical_pair_mask(torch.tensor([0, 1]), torch.tensor([3, 2]), "ACGU").tolist()
        [False, False]
        >>> noncanonical_pair_mask(torch.tensor([0]), torch.tensor([3]), "ACGA").tolist()
        [True]
        >>> noncanonical_pair_mask(torch.tensor([0]), torch.tensor([1]), "GU").tolist()
        [False]

        NumPy input
        >>> import numpy as np
        >>> noncanonical_pair_mask(np.array([0, 1]), np.array([3, 2]), "ACGU").tolist()
        [False, False]
        >>> noncanonical_pair_mask(np.array([0]), np.array([3]), "ACGA").tolist()
        [True]
        >>> noncanonical_pair_mask(np.array([0]), np.array([1]), "GU").tolist()
        [False]

        List input
        >>> noncanonical_pair_mask([0, 1], [3, 2], "ACGU")
        [False, False]
        >>> noncanonical_pair_mask([0], [3], "ACGA")
        [True]
        >>> noncanonical_pair_mask([0], [1], "GU")
        [False]
    """
    if isinstance(pair_i, Tensor) and isinstance(pair_j, Tensor):
        if pair_i.shape != pair_j.shape:
            raise ValueError("pair_i and pair_j must have the same shape")
        if pair_i.device != pair_j.device:
            raise ValueError("pair_i and pair_j must be on the same device")
        return _torch_noncanonical_pair_mask(pair_i, pair_j, sequence, unsafe)
    if isinstance(pair_i, np.ndarray) and isinstance(pair_j, np.ndarray):
        if pair_i.shape != pair_j.shape:
            raise ValueError("pair_i and pair_j must have the same shape")
        return _numpy_noncanonical_pair_mask(pair_i, pair_j, sequence, unsafe)
    if isinstance(pair_i, Sequence) and isinstance(pair_j, Sequence):
        pair_i_np = np.asarray(pair_i, dtype=int)
        pair_j_np = np.asarray(pair_j, dtype=int)
        if pair_i_np.shape != pair_j_np.shape:
            raise ValueError("pair_i and pair_j must have the same shape")
        return _numpy_noncanonical_pair_mask(pair_i_np, pair_j_np, sequence, unsafe).tolist()
    raise TypeError("pair_i and pair_j must be of the same type")


def _torch_noncanonical_pair_mask(pair_i: Tensor, pair_j: Tensor, sequence: str, unsafe: bool) -> Tensor:
    if pair_i.numel() == 0:
        return torch.zeros_like(pair_i, dtype=torch.bool)

    pair_i = pair_i.to(torch.long)
    pair_j = pair_j.to(torch.long)
    flat_i = pair_i.reshape(-1)
    flat_j = pair_j.reshape(-1)
    self_mask = flat_i == flat_j
    has_self = bool(self_mask.any().item())
    if has_self and not unsafe:
        raise ValueError("Self-pairing (i == j) is invalid.")
    if has_self and unsafe:
        warn("Ignoring self-pairing (i == j) in noncanonical_pair_mask.")

    seq_codes = _sequence_codes(sequence)
    if seq_codes.size == 0:
        return torch.zeros_like(pair_i, dtype=torch.bool)
    codes = torch.from_numpy(seq_codes).to(device=pair_i.device, dtype=torch.long)
    codes_i = codes[flat_i]
    codes_j = codes[flat_j]
    unknown = (codes_i < 0) | (codes_j < 0)
    codes_i = codes_i.clamp_min(0)
    codes_j = codes_j.clamp_min(0)
    canonical_table = _canonical_table_torch(pair_i.device)
    canonical = canonical_table[codes_i, codes_j]
    mask = unknown | ~canonical
    if has_self and unsafe:
        mask = mask.masked_fill(self_mask, False)
    return mask.view(pair_i.shape)


def _numpy_noncanonical_pair_mask(pair_i: np.ndarray, pair_j: np.ndarray, sequence: str, unsafe: bool) -> np.ndarray:
    if pair_i.size == 0:
        return np.zeros_like(pair_i, dtype=bool)

    pair_i = pair_i.astype(np.int64, copy=False)
    pair_j = pair_j.astype(np.int64, copy=False)
    flat_i = pair_i.reshape(-1)
    flat_j = pair_j.reshape(-1)
    self_mask = flat_i == flat_j
    has_self = bool(np.any(self_mask))
    if has_self and not unsafe:
        raise ValueError("Self-pairing (i == j) is invalid.")
    if has_self and unsafe:
        warn("Ignoring self-pairing (i == j) in noncanonical_pair_mask.")

    seq_codes = _sequence_codes(sequence)
    if seq_codes.size == 0:
        return np.zeros_like(pair_i, dtype=bool)
    codes_i = seq_codes[flat_i]
    codes_j = seq_codes[flat_j]
    unknown = (codes_i < 0) | (codes_j < 0)
    codes_i = np.clip(codes_i, 0, 3)
    codes_j = np.clip(codes_j, 0, 3)
    canonical = _CANONICAL_TABLE[codes_i, codes_j]
    mask = unknown | ~canonical
    if has_self and unsafe:
        mask[self_mask] = False
    return mask.reshape(pair_i.shape)


def _sequence_codes(sequence: str) -> np.ndarray:
    if not sequence:
        return np.empty((0,), dtype=np.int8)
    seq_bytes = np.frombuffer(sequence.upper().encode("ascii", "replace"), dtype=np.uint8)
    return _BASE_CODE_TABLE[seq_bytes]


def _canonical_table_torch(device: torch.device) -> Tensor:
    table = _CANONICAL_TABLE_TORCH_CACHE.get(device)
    if table is None:
        table = _CANONICAL_TABLE_TORCH.to(device)
        _CANONICAL_TABLE_TORCH_CACHE[device] = table
    return table
