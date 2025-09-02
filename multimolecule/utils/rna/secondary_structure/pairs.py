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

from collections.abc import Sequence
from functools import cached_property
from typing import Dict, List, Tuple, overload

import numpy as np
import torch
from torch import Tensor

from .types import Pair, Pairs, PairsList, Segment, StemSegment


def ensure_pairs_np(pairs: Tensor | np.ndarray | Pairs) -> np.ndarray:
    if isinstance(pairs, Tensor):
        pairs = pairs.detach().cpu().numpy()
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return pairs.astype(int, copy=False)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return pairs.astype(int, copy=False)
    if isinstance(pairs, Sequence):
        if not pairs:
            return np.empty((0, 2), dtype=int)
        pairs = np.asarray(pairs, dtype=int)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return pairs
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def ensure_pairs_list(pairs: Tensor | np.ndarray | Pairs) -> PairsList:
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            return []
        return [(int(i), int(j)) for i, j in pairs.detach().cpu().tolist()]
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            return []
        return [(int(i), int(j)) for i, j in pairs.tolist()]
    return [(int(i), int(j)) for i, j in pairs]


class PairMap:
    def __init__(
        self,
        pairs: Tensor | np.ndarray | Pairs,
        *,
        length: int | None = None,
    ):
        data: Dict[int, int] = {}
        for i, j in ensure_pairs_np(pairs):
            i = int(i)
            j = int(j)
            prev = data.get(i)
            if prev is not None and prev != j:
                raise ValueError(f"position {i} paired to multiple partners")
            prev = data.get(j)
            if prev is not None and prev != i:
                raise ValueError(f"position {j} paired to multiple partners")
            data[i] = j
            data[j] = i
        self._data = data
        self._length = length
        self._list: List[int] | None = None
        if length is not None:
            partner_list = [-1] * length
            for i, j in self._data.items():
                if i < 0 or i >= length:
                    raise ValueError(f"position {i} is out of bounds for length {length}")
                partner_list[i] = int(j)
            self._list = partner_list

    def __len__(self) -> int:
        return len(self._data) // 2

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def get(self, key: int, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key: int) -> int:
        return self._data[key]

    def copy(self) -> PairMap:
        return PairMap(self.pairs, length=self._length)

    def is_loop_linked(self, i_stop1: int, i_start2: int) -> bool:
        prev_idx = i_start2 - 1
        if prev_idx < 0:
            return False
        return i_stop1 + 1 == self._data.get(prev_idx)

    def to_list(self, length: int | None = None) -> List[int]:
        if length is None:
            length = self._length
        if length is None:
            if not self._data:
                return []
            length = max(self._data.keys()) + 1
        if self._list is not None and length == self._length:
            return list(self._list)
        partner_list = [-1] * length
        for i, j in self._data.items():
            if i < 0 or i >= length:
                raise ValueError(f"position {i} is out of bounds for length {length}")
            partner_list[i] = int(j)
        return partner_list

    @cached_property
    def pairs(self) -> List[Pair]:
        if not self._data:
            return []
        pairs = [(int(i), int(j)) for i, j in self._data.items() if i < j]
        pairs.sort()
        return pairs

    @cached_property
    def segments(self) -> List[Segment]:
        if not self._data:
            return []
        pair_i, pair_j, seg_start, seg_len = pairs_to_duplex_segment_arrays(self.pairs)
        segments: List[Segment] = []
        for start, length in zip(seg_start, seg_len):
            start_idx = int(start)
            seg_len = int(length)
            if seg_len <= 0:
                continue
            end_idx = start_idx + seg_len
            segments.append(
                [(int(pi), int(pj)) for pi, pj in zip(pair_i[start_idx:end_idx], pair_j[start_idx:end_idx])]
            )
        return segments


@overload
def normalize_pairs(pairs: Tensor) -> Tensor: ...


@overload
def normalize_pairs(pairs: np.ndarray) -> np.ndarray: ...  # type: ignore[overload-cannot-match]


@overload
def normalize_pairs(pairs: Pairs) -> PairsList: ...  # type: ignore[overload-cannot-match]


def normalize_pairs(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | PairsList:
    """
    Normalize base-pair indices to unique, sorted (i < j) pairs.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Normalized pairs using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> normalize_pairs(torch.tensor([[3, 1], [1, 3]])).tolist()
        [[1, 3]]

        NumPy input
        >>> import numpy as np
        >>> normalize_pairs(np.array([[3, 1], [1, 3]])).tolist()
        [[1, 3]]

        List input
        >>> normalize_pairs([(3, 1), (1, 3), (2, 0)])
        [(0, 2), (1, 3)]
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            return pairs.view(0, 2).to(dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_normalize_pairs(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_normalize_pairs(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.size == 0:
            return []
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return list(map(tuple, _numpy_normalize_pairs(pairs).tolist()))
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_normalize_pairs(pairs: Tensor) -> Tensor:
    pairs = _torch_normalize_pairs_low_high(pairs)
    if pairs.numel() == 0:
        return pairs
    if pairs.shape[0] > 1:
        ii = pairs[:, 0]
        jj = pairs[:, 1]
        ii_prev = ii[:-1]
        ii_next = ii[1:]
        jj_prev = jj[:-1]
        jj_next = jj[1:]
        is_sorted = torch.all((ii_next > ii_prev) | ((ii_next == ii_prev) & (jj_next >= jj_prev))).item()
        if is_sorted:
            has_dups = torch.any((ii_next == ii_prev) & (jj_next == jj_prev)).item()
            if not has_dups:
                return pairs
            return torch.unique_consecutive(pairs, dim=0)
    pairs = torch.unique(pairs, dim=0)
    if pairs.numel() == 0:
        return pairs
    key = pairs[:, 0] * (int(pairs[:, 1].max().item()) + 1 if pairs.numel() else 1) + pairs[:, 1]
    order = torch.argsort(key)
    return pairs[order]


def _torch_normalize_pairs_low_high(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0:
        return pairs.view(0, 2).to(dtype=torch.long)
    low = torch.minimum(pairs[:, 0], pairs[:, 1]).to(torch.long)
    high = torch.maximum(pairs[:, 0], pairs[:, 1]).to(torch.long)
    return torch.stack([low, high], dim=1)


def _numpy_normalize_pairs(pairs: np.ndarray) -> np.ndarray:
    pairs = _numpy_normalize_pairs_low_high(pairs)
    if pairs.size == 0:
        return pairs
    pairs = np.unique(pairs, axis=0)
    if pairs.size == 0:
        return pairs
    ord_idx = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[ord_idx]


def _numpy_normalize_pairs_low_high(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.column_stack((low, high)).astype(int, copy=False)


def _torch_compress_endpoints(start_i: Tensor, start_j: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
    positions = torch.unique(torch.cat([start_i, start_j]), sorted=True)
    positions = positions.to(dtype=torch.long)
    comp_len = int(positions.numel())
    if comp_len == 0:
        empty = start_i.new_empty((0,), dtype=torch.long)
        return positions, empty, empty, 0
    ci = torch.searchsorted(positions, start_i)
    cj = torch.searchsorted(positions, start_j)
    return positions, ci, cj, comp_len


def _numpy_compress_endpoints(
    start_i: np.ndarray, start_j: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    positions = np.unique(np.concatenate([start_i, start_j]).astype(int, copy=False))
    comp_len = int(positions.size)
    if comp_len == 0:
        empty = np.empty((0,), dtype=int)
        return positions, empty, empty, 0
    ci = np.searchsorted(positions, start_i)
    cj = np.searchsorted(positions, start_j)
    return positions.astype(int, copy=False), ci.astype(int, copy=False), cj.astype(int, copy=False), comp_len


def _torch_endpoint_prefix(cj: Tensor, comp_len: int) -> Tuple[Tensor, Tensor]:
    counts = torch.bincount(cj, minlength=comp_len)
    prefix = torch.cumsum(counts, dim=0)
    return counts, prefix


def _numpy_endpoint_prefix(cj: np.ndarray, comp_len: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(cj, minlength=comp_len)
    prefix = counts.cumsum()
    return counts.astype(int, copy=False), prefix.astype(int, copy=False)


def _torch_partner_indices(ci: Tensor, cj: Tensor, comp_len: int) -> Tensor:
    partner_idx = ci.new_empty((comp_len,), dtype=torch.long)
    partner_idx[ci] = cj
    partner_idx[cj] = ci
    return partner_idx


def _numpy_partner_indices(ci: np.ndarray, cj: np.ndarray, comp_len: int) -> np.ndarray:
    partner_idx = np.empty(comp_len, dtype=int)
    partner_idx[ci] = cj
    partner_idx[cj] = ci
    return partner_idx.astype(int, copy=False)


def sort_pairs(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | PairsList:
    """
    Sort base-pair indices by (i, j) without normalization or de-duplication.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Sorted pairs using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> sort_pairs(torch.tensor([[2, 5], [0, 1]])).tolist()
        [[0, 1], [2, 5]]

        NumPy input
        >>> import numpy as np
        >>> sort_pairs(np.array([[2, 5], [0, 1]])).tolist()
        [[0, 1], [2, 5]]

        List input
        >>> sort_pairs([(2, 5), (0, 1)])
        [(0, 1), (2, 5)]
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            return pairs.view(0, 2)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_sort_pairs(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_sort_pairs(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return list(map(tuple, _numpy_sort_pairs(pairs).tolist()))
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_sort_pairs(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0 or pairs.shape[0] < 2:
        return pairs
    ii = pairs[:, 0]
    jj = pairs[:, 1]
    ii_prev = ii[:-1]
    ii_next = ii[1:]
    jj_prev = jj[:-1]
    jj_next = jj[1:]
    is_sorted = torch.all((ii_next > ii_prev) | ((ii_next == ii_prev) & (jj_next >= jj_prev))).item()
    if is_sorted:
        return pairs
    max_j = int(jj.max().item()) if jj.numel() else 0
    key = ii * (max_j + 1) + jj
    order = torch.argsort(key)
    return pairs[order]


def _numpy_sort_pairs(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0 or pairs.shape[0] < 2:
        return pairs
    ii = pairs[:, 0]
    jj = pairs[:, 1]
    ii_prev = ii[:-1]
    ii_next = ii[1:]
    jj_prev = jj[:-1]
    jj_next = jj[1:]
    is_sorted = np.all((ii_next > ii_prev) | ((ii_next == ii_prev) & (jj_next >= jj_prev)))
    if is_sorted:
        return pairs
    order = np.lexsort((jj, ii))
    return pairs[order]


@overload
def pairs_to_duplex_segment_arrays(pairs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


@overload
def pairs_to_duplex_segment_arrays(  # type: ignore[overload-cannot-match]
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def pairs_to_duplex_segment_arrays(  # type: ignore[overload-cannot-match]
    pairs: Pairs,
) -> Tuple[List[int], List[int], List[int], List[int]]: ...


def pairs_to_duplex_segment_arrays(
    pairs: Tensor | np.ndarray | Pairs,
) -> (
    Tuple[Tensor, Tensor, Tensor, Tensor]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # noqa: W503
    | Tuple[List[int], List[int], List[int], List[int]]  # noqa: W503
):
    """
    Convert base pairs to bulge-tolerant duplex segments while preserving actual pairs.

    Returns (pair_i, pair_j, segment_starts, segment_lengths), where segment_starts/segment_lengths
    index into the flattened pair arrays.
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            empty = pairs.new_empty((0,), dtype=torch.long)
            return empty, empty, empty, empty
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_pairs_to_duplex_segment_arrays(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr, empty_arr
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_pairs_to_duplex_segment_arrays(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return [], [], [], []
        pairs_array = np.asarray(pairs, dtype=int)
        if pairs_array.size == 0:
            return [], [], [], []
        if pairs_array.ndim != 2 or pairs_array.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        pair_i, pair_j, seg_start, seg_len = _numpy_pairs_to_duplex_segment_arrays(pairs_array)
        return pair_i.tolist(), pair_j.tolist(), seg_start.tolist(), seg_len.tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_pairs_to_duplex_segment_arrays(pairs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if pairs.numel() == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
    pairs = _torch_normalize_pairs(pairs)
    if pairs.numel() == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty

    i = pairs[:, 0].contiguous()
    j = pairs[:, 1].contiguous()
    positions, ci, cj, comp_len = _torch_compress_endpoints(i, j)
    if comp_len == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty
    partner_idx = _torch_partner_indices(ci, cj, comp_len)
    partner_pos = positions[partner_idx]

    left_mask = positions < partner_pos
    if not bool(left_mask.any().item()):
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty

    left_indices = torch.nonzero(left_mask, as_tuple=False).view(-1)
    pair_i = positions[left_indices]
    pair_j = partner_pos[left_indices]

    prev_idx = partner_idx[:-1] - 1
    valid_prev = partner_idx[:-1] > 0
    prev_pos = torch.full_like(prev_idx, -1)
    if prev_pos.numel() > 0:
        prev_pos[valid_prev] = positions[prev_idx[valid_prev]]
    next_pos = positions[1:]
    link = left_mask[:-1] & valid_prev & (partner_pos[1:] == prev_pos) & (next_pos < prev_pos)

    prev_link = torch.zeros_like(left_mask, dtype=torch.bool)
    if link.numel() > 0:
        prev_link[1:] = link
    start_mask = left_mask & ~prev_link
    start_idx = torch.nonzero(start_mask, as_tuple=False).view(-1)
    if start_idx.numel() == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, empty

    if link.numel() == 0:
        lengths = torch.ones_like(start_idx)
    else:
        breaks = torch.nonzero(~link, as_tuple=False).view(-1)
        sentinel = torch.full((1,), positions.numel() - 1, device=positions.device, dtype=breaks.dtype)
        if breaks.numel() == 0:
            breaks = sentinel
        else:
            breaks = torch.cat([breaks, sentinel])
        break_idx = torch.searchsorted(breaks, start_idx)
        end_idx = breaks[break_idx]
        lengths = end_idx - start_idx + 1

    left_rank = torch.full((positions.numel(),), -1, dtype=torch.long, device=positions.device)
    left_rank[left_indices] = torch.arange(left_indices.numel(), device=positions.device)
    seg_start = left_rank[start_idx]
    return pair_i, pair_j, seg_start, lengths


def _numpy_pairs_to_duplex_segment_arrays(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty, empty
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
    pairs = _numpy_normalize_pairs(pairs)
    if pairs.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty, empty
    i = pairs[:, 0]
    j = pairs[:, 1]
    positions, ci, cj, comp_len = _numpy_compress_endpoints(i, j)
    if comp_len == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty, empty
    partner_idx = _numpy_partner_indices(ci, cj, comp_len)
    partner_pos = positions[partner_idx]

    left_mask = positions < partner_pos
    if not np.any(left_mask):
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty, empty

    left_indices = np.flatnonzero(left_mask)
    pair_i = positions[left_indices]
    pair_j = partner_pos[left_indices]

    prev_idx = partner_idx[:-1] - 1
    valid_prev = partner_idx[:-1] > 0
    prev_pos = np.full(prev_idx.shape, -1, dtype=int)
    if prev_pos.size:
        prev_pos[valid_prev] = positions[prev_idx[valid_prev]]
    next_pos = positions[1:]
    link = left_mask[:-1] & valid_prev & (partner_pos[1:] == prev_pos) & (next_pos < prev_pos)

    prev_link = np.zeros_like(left_mask, dtype=bool)
    if link.size:
        prev_link[1:] = link
    start_mask = left_mask & ~prev_link
    start_idx = np.flatnonzero(start_mask)
    if start_idx.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty, empty

    if link.size == 0:
        lengths = np.ones_like(start_idx)
    else:
        breaks = np.flatnonzero(~link)
        sentinel = np.array([positions.size - 1], dtype=breaks.dtype)
        if breaks.size == 0:
            breaks = sentinel
        else:
            breaks = np.concatenate([breaks, sentinel])
        break_idx = np.searchsorted(breaks, start_idx, side="left")
        end_idx = breaks[break_idx]
        lengths = end_idx - start_idx + 1

    left_rank = np.full(positions.size, -1, dtype=int)
    left_rank[left_indices] = np.arange(left_indices.size, dtype=int)
    seg_start = left_rank[start_idx]
    return (
        pair_i.astype(int, copy=False),
        pair_j.astype(int, copy=False),
        seg_start.astype(int, copy=False),
        lengths.astype(int, copy=False),
    )


@overload
def pairs_to_helix_segment_arrays(pairs: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...


@overload
def pairs_to_helix_segment_arrays(  # type: ignore[overload-cannot-match]
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def pairs_to_helix_segment_arrays(  # type: ignore[overload-cannot-match]
    pairs: Pairs,
) -> Tuple[List[int], List[int], List[int]]: ...


def pairs_to_helix_segment_arrays(
    pairs: Tensor | np.ndarray | Pairs,
) -> Tuple[Tensor, Tensor, Tensor] | Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[List[int], List[int], List[int]]:
    """
    Convert base pairs to strict stacked segments (no bulge loops).
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            empty = pairs.new_empty((0,), dtype=torch.long)
            return empty, empty, empty
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_pairs_to_helix_segment_arrays(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_pairs_to_helix_segment_arrays(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr
        pairs_array = np.asarray(pairs, dtype=int)
        if pairs_array.size == 0:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr
        if pairs_array.ndim != 2 or pairs_array.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        start_i, start_j, lengths = _numpy_pairs_to_helix_segment_arrays(pairs_array)
        return start_i.tolist(), start_j.tolist(), lengths.tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_pairs_to_helix_segment_arrays(norm: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    if norm.numel() == 0:
        empty = norm.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    pairs = _torch_normalize_pairs(norm)
    if pairs.numel() == 0:
        empty = norm.new_empty((0,), dtype=torch.long)
        return empty, empty, empty

    i = pairs[:, 0].to(torch.long)
    j = pairs[:, 1].to(torch.long)
    if i.numel() == 0:
        empty = norm.new_empty((0,), dtype=torch.long)
        return empty, empty, empty

    # Runs where (i, j) increase by (+1, -1) indicate helix continuation
    cont = (i[1:] == i[:-1] + 1) & (j[1:] == j[:-1] - 1)
    start_mask = torch.cat([torch.tensor([True], device=i.device), ~cont])
    start_indices = torch.nonzero(start_mask, as_tuple=False).view(-1)
    if start_indices.numel() == 0:
        empty = norm.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    end_indices = torch.cat([start_indices[1:], torch.tensor([i.numel()], device=i.device, dtype=start_indices.dtype)])
    lengths = end_indices - start_indices

    start_i = i[start_indices]
    start_j = j[start_indices]
    return start_i, start_j, lengths


def _numpy_pairs_to_helix_segment_arrays(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
    pairs = _numpy_normalize_pairs(pairs)
    if pairs.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty

    i = pairs[:, 0].astype(int, copy=False)
    j = pairs[:, 1].astype(int, copy=False)
    cont = (i[1:] == i[:-1] + 1) & (j[1:] == j[:-1] - 1)
    start_mask = np.concatenate([[True], ~cont])
    start_indices = np.flatnonzero(start_mask)
    if start_indices.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty
    end_indices = np.concatenate([start_indices[1:], np.array([i.size], dtype=int)])
    lengths = end_indices - start_indices

    start_i = i[start_indices]
    start_j = j[start_indices]
    return start_i.astype(int, copy=False), start_j.astype(int, copy=False), lengths.astype(int, copy=False)


@overload
def pairs_to_stem_segment_arrays(pairs: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...


@overload
def pairs_to_stem_segment_arrays(  # type: ignore[overload-cannot-match]
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def pairs_to_stem_segment_arrays(  # type: ignore[overload-cannot-match]
    pairs: Pairs,
) -> Tuple[List[int], List[int], List[int]]: ...


def pairs_to_stem_segment_arrays(
    pairs: Tensor | np.ndarray | Pairs,
) -> Tuple[Tensor, Tensor, Tensor] | Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[List[int], List[int], List[int]]:
    """
    Convert base pairs to stem segments that allow bulge/internal loops.
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            empty = pairs.new_empty((0,), dtype=torch.long)
            return empty, empty, empty
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_pairs_to_stem_segment_arrays(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_pairs_to_stem_segment_arrays(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr
        pairs_array = np.asarray(pairs, dtype=int)
        if pairs_array.size == 0:
            empty_arr = np.empty((0,), dtype=int)
            return empty_arr, empty_arr, empty_arr
        if pairs_array.ndim != 2 or pairs_array.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        start_i, start_j, lengths = _numpy_pairs_to_stem_segment_arrays(pairs_array)
        return start_i.tolist(), start_j.tolist(), lengths.tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def stem_segment_arrays_to_stem_segment_list(
    start_i: Tensor | np.ndarray | Sequence[int],
    start_j: Tensor | np.ndarray | Sequence[int],
    lengths: Tensor | np.ndarray | Sequence[int],
    tier: int = 0,
) -> List[StemSegment]:
    stem_segments: List[StemSegment] = []
    if hasattr(start_i, "numel"):
        count = int(start_i.numel())
    elif hasattr(start_i, "size"):
        count = int(start_i.size)
    else:
        count = len(start_i)
    for idx in range(count):
        seg_len = int(lengths[idx])
        if seg_len <= 0:
            continue
        start_5p = int(start_i[idx])
        start_3p = int(start_j[idx])
        stop_5p = start_5p + seg_len - 1
        stop_3p = start_3p - seg_len + 1
        stem_segments.append(StemSegment(start_5p, stop_5p, start_3p, stop_3p, tier))
    return stem_segments


def _torch_pairs_to_stem_segment_arrays(norm: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    if norm.numel() == 0:
        empty = norm.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    pairs = _torch_normalize_pairs(norm)
    if pairs.numel() == 0:
        empty = norm.new_empty((0,), dtype=torch.long)
        return empty, empty, empty

    i = pairs[:, 0].contiguous()
    j = pairs[:, 1].contiguous()
    positions, ci, cj, comp_len = _torch_compress_endpoints(i, j)
    if comp_len == 0:
        empty = pairs.new_empty((0,), dtype=torch.long)
        return empty, empty, empty

    partner_idx = _torch_partner_indices(ci, cj, comp_len)
    partners = positions[partner_idx]
    left_mask = positions < partners
    if positions.numel() == 1:
        start_idx = torch.nonzero(left_mask, as_tuple=False).view(-1)
        if start_idx.numel() == 0:
            empty = pairs.new_empty((0,), dtype=torch.long)
            return empty, empty, empty
        lengths = torch.ones_like(start_idx)
    else:
        next_pos = positions[1:]
        next_partner = partners[1:]
        partner_idx_k = partner_idx[:-1]
        valid_prev = partner_idx_k > 0
        prev_idx = torch.clamp(partner_idx_k - 1, min=0)
        prev_partner_pos = torch.where(valid_prev, positions[prev_idx], torch.full_like(prev_idx, -1))

        link = left_mask[:-1] & valid_prev & (next_partner == prev_partner_pos) & (next_pos < prev_partner_pos)
        prev_link = torch.zeros_like(left_mask, dtype=torch.bool)
        if link.numel() > 0:
            prev_link[1:] = link
        starts_mask = left_mask & ~prev_link
        start_idx = torch.nonzero(starts_mask, as_tuple=False).view(-1)
        if start_idx.numel() == 0:
            empty = pairs.new_empty((0,), dtype=torch.long)
            return empty, empty, empty
        if link.numel() == 0:
            lengths = torch.ones_like(start_idx)
        else:
            breaks = torch.nonzero(~link, as_tuple=False).view(-1)
            sentinel = torch.full((1,), positions.numel() - 1, device=positions.device, dtype=breaks.dtype)
            if breaks.numel() == 0:
                breaks = sentinel
            else:
                breaks = torch.cat([breaks, sentinel])
            break_idx = torch.searchsorted(breaks, start_idx)
            first_break = breaks[break_idx]
            lengths = first_break - start_idx + 1

    start_i = positions[start_idx]
    start_j = partners[start_idx]
    return start_i, start_j, lengths


def _numpy_pairs_to_stem_segment_arrays(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
    pairs = _numpy_normalize_pairs(pairs)
    if pairs.size == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty

    i = pairs[:, 0]
    j = pairs[:, 1]
    positions, ci, cj, comp_len = _numpy_compress_endpoints(i, j)
    if comp_len == 0:
        empty = np.empty((0,), dtype=int)
        return empty, empty, empty

    partner_idx = _numpy_partner_indices(ci, cj, comp_len)
    partners = positions[partner_idx]
    left_mask = positions < partners
    if positions.size == 1:
        start_idx = np.flatnonzero(left_mask)
        if start_idx.size == 0:
            empty = np.empty((0,), dtype=int)
            return empty, empty, empty
        lengths = np.ones_like(start_idx)
    else:
        next_pos = positions[1:]
        next_partner = partners[1:]
        partner_idx_k = partner_idx[:-1]
        valid_prev = partner_idx_k > 0
        prev_idx = np.maximum(partner_idx_k - 1, 0)
        prev_partner_pos = np.where(valid_prev, positions[prev_idx], -1)

        link = left_mask[:-1] & valid_prev & (next_partner == prev_partner_pos) & (next_pos < prev_partner_pos)
        prev_link = np.zeros_like(left_mask, dtype=bool)
        if link.size > 0:
            prev_link[1:] = link
        starts_mask = left_mask & ~prev_link
        start_idx = np.flatnonzero(starts_mask)
        if start_idx.size == 0:
            empty = np.empty((0,), dtype=int)
            return empty, empty, empty
        if link.size == 0:
            lengths = np.ones_like(start_idx)
        else:
            breaks = np.flatnonzero(~link)
            sentinel = np.array([positions.size - 1], dtype=breaks.dtype)
            if breaks.size == 0:
                breaks = sentinel
            else:
                breaks = np.concatenate([breaks, sentinel])
            break_idx = np.searchsorted(breaks, start_idx, side="left")
            first_break = breaks[break_idx]
            lengths = first_break - start_idx + 1

    start_i = positions[start_idx]
    start_j = partners[start_idx]
    return start_i.astype(int, copy=False), start_j.astype(int, copy=False), lengths.astype(int, copy=False)


def _torch_segment_mask(mask: Tensor | np.ndarray | Sequence[int] | None, size: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones((size,), dtype=torch.bool, device=device)
    if isinstance(mask, Tensor):
        if mask.dtype == torch.bool:
            return mask
        mask_tensor = torch.zeros((size,), dtype=torch.bool, device=device)
        if mask.numel() > 0:
            mask_tensor[mask.to(dtype=torch.long)] = True
        return mask_tensor
    if isinstance(mask, np.ndarray):
        if mask.dtype == bool:
            return torch.tensor(mask, dtype=torch.bool, device=device)
        mask_tensor = torch.zeros((size,), dtype=torch.bool, device=device)
        if mask.size > 0:
            mask_tensor[torch.as_tensor(mask.astype(int, copy=False), dtype=torch.long, device=device)] = True
        return mask_tensor
    if isinstance(mask, (list, tuple)):
        if mask and isinstance(mask[0], bool):
            return torch.tensor(mask, dtype=torch.bool, device=device)
        mask_tensor = torch.zeros((size,), dtype=torch.bool, device=device)
        if mask:
            mask_tensor[torch.tensor(mask, dtype=torch.long, device=device)] = True
        return mask_tensor
    raise TypeError("mask must be a boolean array or sequence of indices")


def _numpy_segment_mask(mask: Tensor | np.ndarray | Sequence[int] | None, size: int) -> np.ndarray:
    if mask is None:
        return np.ones(size, dtype=bool)
    if isinstance(mask, np.ndarray) and mask.dtype == bool:
        return mask
    if isinstance(mask, (list, tuple)) and mask and isinstance(mask[0], bool):
        return np.asarray(mask, dtype=bool)
    if isinstance(mask, (list, tuple, np.ndarray)):
        mask_array = np.zeros(size, dtype=bool)
        if size > 0:
            mask_array[np.asarray(mask, dtype=int)] = True
        return mask_array
    raise TypeError("mask must be a boolean array or sequence of indices")


def segment_arrays_to_pairs(
    segments: Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor] | List[Segment],
    mask: Tensor | np.ndarray | Sequence[int] | None = None,
    empty: np.ndarray | None = None,
) -> Tensor | np.ndarray:
    """
    Convert segments back to base pairs.
    """
    if isinstance(segments, tuple):
        if len(segments) == 3:
            start_i, start_j, lengths = segments
            if isinstance(start_i, Tensor):
                mask_tensor = _torch_segment_mask(mask, int(start_i.numel()), start_i.device)
                return _torch_segment_arrays_to_pairs(start_i, start_j, lengths, mask_tensor)
            if isinstance(start_i, np.ndarray):
                if empty is None:
                    empty = np.empty((0, 2), dtype=int)
                mask_array = _numpy_segment_mask(mask, int(start_i.size))
                return _numpy_segment_arrays_to_pairs(start_i, start_j, lengths, mask_array, empty)
            if isinstance(start_i, (list, tuple)):
                start_i = np.asarray(start_i, dtype=int)
                start_j = np.asarray(start_j, dtype=int)
                lengths = np.asarray(lengths, dtype=int)
                if empty is None:
                    empty = np.empty((0, 2), dtype=int)
                mask_array = _numpy_segment_mask(mask, int(start_i.size))
                return _numpy_segment_arrays_to_pairs(start_i, start_j, lengths, mask_array, empty)
            raise TypeError("segments tuple must contain torch.Tensor, numpy.ndarray, or list values")
        if len(segments) == 4:
            pair_i, pair_j, seg_start, seg_len = segments
            if isinstance(pair_i, Tensor):
                mask_tensor = _torch_segment_mask(mask, int(seg_len.numel()), pair_i.device)
                return _torch_duplex_segment_arrays_to_pairs(pair_i, pair_j, seg_start, seg_len, mask_tensor)
            if isinstance(pair_i, np.ndarray):
                if empty is None:
                    empty = np.empty((0, 2), dtype=int)
                mask_array = _numpy_segment_mask(mask, int(seg_len.size))
                return _numpy_duplex_segment_arrays_to_pairs(pair_i, pair_j, seg_start, seg_len, mask_array, empty)
            if isinstance(pair_i, (list, tuple)):
                pair_i = np.asarray(pair_i, dtype=int)
                pair_j = np.asarray(pair_j, dtype=int)
                seg_start = np.asarray(seg_start, dtype=int)
                seg_len = np.asarray(seg_len, dtype=int)
                if empty is None:
                    empty = np.empty((0, 2), dtype=int)
                mask_array = _numpy_segment_mask(mask, int(seg_len.size))
                return _numpy_duplex_segment_arrays_to_pairs(pair_i, pair_j, seg_start, seg_len, mask_array, empty)
            raise TypeError("segments tuple must contain torch.Tensor, numpy.ndarray, or list values")
        raise ValueError("segments must be (start_i, start_j, lengths) or (pair_i, pair_j, seg_start, seg_len)")
    assert not isinstance(segments, tuple)
    if empty is None:
        empty = np.empty((0, 2), dtype=int)
    if mask is not None:
        if (isinstance(mask, np.ndarray) and mask.dtype == bool) or (
            isinstance(mask, (list, tuple)) and mask and isinstance(mask[0], bool)
        ):
            segments = [seg for seg, keep in zip(segments, mask) if keep]
        elif isinstance(mask, (list, tuple, np.ndarray)):
            segments = [segments[int(idx)] for idx in mask]
        else:
            raise TypeError("mask must be a boolean array or sequence of indices")
    return segment_list_to_pairs(segments, empty)


def _torch_segment_arrays_to_pairs(start_i: Tensor, start_j: Tensor, lengths: Tensor, mask: Tensor) -> Tensor:
    if mask.numel() == 0 or not bool(mask.any().item()):
        return start_i.new_empty((0, 2), dtype=torch.long)
    lengths = lengths.to(dtype=torch.long)
    valid = mask.to(dtype=torch.bool) & (lengths > 0)
    if not bool(valid.any().item()):
        return start_i.new_empty((0, 2), dtype=torch.long)
    indices = torch.nonzero(valid, as_tuple=False).view(-1)
    seg_lengths = lengths[indices]
    total = int(seg_lengths.sum().item())
    if total <= 0:
        return start_i.new_empty((0, 2), dtype=torch.long)
    seg_ids = torch.repeat_interleave(torch.arange(indices.numel(), device=start_i.device), seg_lengths)
    seg_starts = torch.cumsum(seg_lengths, dim=0) - seg_lengths
    offsets = torch.arange(total, device=start_i.device) - torch.repeat_interleave(seg_starts, seg_lengths)
    start_i_rep = start_i[indices][seg_ids].to(dtype=torch.long)
    start_j_rep = start_j[indices][seg_ids].to(dtype=torch.long)
    out = torch.stack([start_i_rep + offsets, start_j_rep - offsets], dim=1)
    key = out[:, 0] * (int(out[:, 1].max().item()) + 1) + out[:, 1]
    order = torch.argsort(key)
    return out[order]


def _torch_duplex_segment_arrays_to_pairs(
    pair_i: Tensor,
    pair_j: Tensor,
    seg_start: Tensor,
    seg_len: Tensor,
    mask: Tensor,
) -> Tensor:
    if mask.numel() == 0 or not bool(mask.any().item()):
        return pair_i.new_empty((0, 2), dtype=torch.long)
    seg_len = seg_len.to(dtype=torch.long)
    valid = mask.to(dtype=torch.bool) & (seg_len > 0)
    if not bool(valid.any().item()):
        return pair_i.new_empty((0, 2), dtype=torch.long)
    indices = torch.nonzero(valid, as_tuple=False).view(-1)
    seg_lengths = seg_len[indices]
    total = int(seg_lengths.sum().item())
    if total <= 0:
        return pair_i.new_empty((0, 2), dtype=torch.long)
    seg_ids = torch.repeat_interleave(torch.arange(indices.numel(), device=pair_i.device), seg_lengths)
    seg_starts = torch.cumsum(seg_lengths, dim=0) - seg_lengths
    offsets = torch.arange(total, device=pair_i.device) - torch.repeat_interleave(seg_starts, seg_lengths)
    pair_start_rep = seg_start[indices][seg_ids].to(dtype=torch.long)
    pair_idx = pair_start_rep + offsets
    out = torch.stack([pair_i[pair_idx], pair_j[pair_idx]], dim=1)
    key = out[:, 0] * (int(out[:, 1].max().item()) + 1 if out.numel() else 1) + out[:, 1]
    order = torch.argsort(key)
    return out[order]


def _numpy_segment_arrays_to_pairs(
    start_i: np.ndarray,
    start_j: np.ndarray,
    lengths: np.ndarray,
    mask: np.ndarray,
    empty: np.ndarray,
) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return empty
    lengths = lengths.astype(int, copy=False)
    valid = mask.astype(bool, copy=False) & (lengths > 0)
    if not np.any(valid):
        return empty
    indices = np.flatnonzero(valid)
    seg_lengths = lengths[indices].astype(int, copy=False)
    total = int(seg_lengths.sum())
    if total <= 0:
        return empty
    seg_ids = np.repeat(np.arange(indices.size, dtype=int), seg_lengths)
    seg_starts = np.cumsum(seg_lengths) - seg_lengths
    offsets = np.arange(total, dtype=int) - np.repeat(seg_starts, seg_lengths)
    start_i_rep = start_i[indices][seg_ids]
    start_j_rep = start_j[indices][seg_ids]
    out = np.stack((start_i_rep + offsets, start_j_rep - offsets), axis=1)
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _numpy_duplex_segment_arrays_to_pairs(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    seg_start: np.ndarray,
    seg_len: np.ndarray,
    mask: np.ndarray,
    empty: np.ndarray,
) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return empty
    seg_len = seg_len.astype(int, copy=False)
    valid = mask.astype(bool, copy=False) & (seg_len > 0)
    if not np.any(valid):
        return empty
    indices = np.flatnonzero(valid)
    seg_lengths = seg_len[indices].astype(int, copy=False)
    total = int(seg_lengths.sum())
    if total <= 0:
        return empty
    seg_ids = np.repeat(np.arange(indices.size, dtype=int), seg_lengths)
    seg_starts = np.cumsum(seg_lengths) - seg_lengths
    offsets = np.arange(total, dtype=int) - np.repeat(seg_starts, seg_lengths)
    pair_start_rep = seg_start[indices][seg_ids].astype(int, copy=False)
    pair_idx = pair_start_rep + offsets
    out = np.stack((pair_i[pair_idx], pair_j[pair_idx]), axis=1)
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def segment_list_to_pairs(segments: List[Segment], empty: np.ndarray) -> np.ndarray:
    if not segments:
        return empty
    pairs = np.array([pair for seg in segments for pair in seg], dtype=int)
    if pairs.size == 0:
        return empty
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]
