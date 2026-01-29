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
from typing import Dict, List, Tuple, overload

import numpy as np
import torch
from torch import Tensor

from ...environment import env_int
from .pairs import (
    _numpy_compress_endpoints,
    _numpy_endpoint_prefix,
    _numpy_pairs_to_duplex_segment_arrays,
    _numpy_pairs_to_stem_segment_arrays,
    _numpy_sort_pairs,
    _torch_compress_endpoints,
    _torch_endpoint_prefix,
    _torch_pairs_to_duplex_segment_arrays,
    _torch_pairs_to_stem_segment_arrays,
    _torch_sort_pairs,
    segment_arrays_to_pairs,
)
from .types import Pair, Pairs, PairsList, Tiers

_CROSSING_N2_THRESHOLD = env_int("CROSSING_N2_THRESHOLD", 2048)


@overload
def split_pseudoknot_pairs(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...


@overload
def split_pseudoknot_pairs(pairs: Pairs) -> Tuple[PairsList, PairsList]: ...  # type: ignore[overload-cannot-match]


@overload
def split_pseudoknot_pairs(pairs: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore[overload-cannot-match]


def split_pseudoknot_pairs(
    pairs: Tensor | np.ndarray | Pairs,
) -> Tuple[Tensor | np.ndarray | PairsList, Tensor | np.ndarray | PairsList]:
    """
    Split base pairs into primary and pseudoknot pairs using segment-level MWIS.

    Pairs are expected to be normalized (unique, sorted with i < j).
    Use ``normalize_pairs`` if you need to normalize raw inputs.

    Tie-breaks order for equal total base pairs is lexicographic on:

    1. minimize unpaired-within-span
    2. minimize total span
    3. minimize number of segments
    4. deterministic segment order (prefer 3' segments)

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        (nested_pairs, pseudoknot_pairs) using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> primary, pseudoknot_pairs = split_pseudoknot_pairs(torch.tensor([[0, 2], [1, 3]]))
        >>> primary.tolist(), pseudoknot_pairs.tolist()
        ([[1, 3]], [[0, 2]])

        NumPy input
        >>> import numpy as np
        >>> primary, pseudoknot_pairs = split_pseudoknot_pairs(np.array([[0, 2], [1, 3]]))
        >>> primary.tolist(), pseudoknot_pairs.tolist()
        ([[1, 3]], [[0, 2]])

        List input
        >>> split_pseudoknot_pairs([(0, 2), (1, 3)])
        ([(1, 3)], [(0, 2)])
        >>> split_pseudoknot_pairs([(0, 3), (1, 2)])
        ([(0, 3), (1, 2)], [])
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        primary, pseudoknot = _torch_split_pseudoknot_pairs(pairs)
        return _torch_sort_pairs(primary), _torch_sort_pairs(pseudoknot)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        primary, pseudoknot = _numpy_split_pseudoknot_pairs(pairs)
        return _numpy_sort_pairs(primary), _numpy_sort_pairs(pseudoknot)
    if isinstance(pairs, Sequence):
        if not pairs:
            return [], []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        primary, pseudoknot = _numpy_split_pseudoknot_pairs(pairs)
        primary = _numpy_sort_pairs(primary)
        pseudoknot = _numpy_sort_pairs(pseudoknot)
        return list(map(tuple, primary.tolist())), list(map(tuple, pseudoknot.tolist()))
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_split_pseudoknot_pairs(pairs: Tensor) -> Tuple[Tensor, Tensor]:
    if pairs.numel() == 0:
        empty = pairs.view(0, 2)
        return empty, empty
    crossing_mask = _torch_crossing_mask(pairs)
    if not bool(crossing_mask.any().item()):
        return pairs, pairs.new_empty((0, 2))
    empty = pairs[:0]
    if pairs.shape[0] < 2:
        return pairs, empty
    device = pairs.device
    pair_i, pair_j, seg_start, seg_len = _torch_pairs_to_duplex_segment_arrays(pairs)
    if seg_len.numel() == 0:
        return pairs, empty
    start_i = pair_i[seg_start]
    start_j = pair_j[seg_start]
    nested_idx = _torch_mwis_segments(start_i, start_j, seg_len)
    if nested_idx.numel() == 0:
        return empty, pairs
    seg_mask = torch.zeros_like(seg_len, dtype=torch.bool, device=device)
    seg_mask[nested_idx] = True
    primary = segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=seg_mask)
    pseudoknot = segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=~seg_mask)
    return primary, pseudoknot


def _torch_mwis_segments(start_i: Tensor, start_j: Tensor, lengths: Tensor) -> Tensor:
    if start_i.numel() == 0:
        return start_i.new_empty((0,), dtype=torch.long)
    start_i = start_i.contiguous()
    start_j = start_j.contiguous()
    _, ci, cj, comp_len = _torch_compress_endpoints(start_i, start_j)
    if comp_len <= 1:
        return start_i.new_empty((0,), dtype=torch.long)

    span = (start_j - start_i).to(torch.long)
    unpaired = span + 1 - 2 * lengths.to(torch.long)
    comp_vals = torch.stack(
        [
            lengths.to(torch.long),
            (-unpaired).to(torch.long),
            (-span).to(torch.long),
            -torch.ones_like(lengths, dtype=torch.long),
        ],
        dim=1,
    )
    seg_count = int(start_i.numel())
    seg_ids = torch.arange(seg_count, device=start_i.device, dtype=torch.long)
    # Stable sort by end position, then original index to preserve tie-break order
    order = torch.argsort(cj * (seg_count + 1) + seg_ids)
    ci_sorted = ci[order]
    cj_sorted = cj[order]
    comp_sorted = comp_vals[order].T  # (comp_count, seg_count)
    _, prefix = _torch_endpoint_prefix(cj_sorted, comp_len)

    selected = _torch_mwis_select(ci_sorted, order, prefix, comp_sorted, comp_len)
    return selected


def _torch_mwis_select(
    ci_sorted: Tensor,
    order: Tensor,
    prefix: Tensor,
    comp_sorted: Tensor,
    comp_len: int,
) -> Tensor:
    comp_count = int(comp_sorted.shape[0])
    device = ci_sorted.device
    dp = torch.zeros((comp_count, comp_len, comp_len), dtype=torch.long, device=device)
    choice = torch.full((comp_len, comp_len), -1, dtype=torch.long, device=device)

    for j in range(1, comp_len):
        best = dp[:, :j, j - 1].clone()
        best_idx = torch.full((j,), -1, dtype=torch.long, device=device)
        start = int(prefix[j - 1].item()) if j > 0 else 0
        end = int(prefix[j].item())
        for cand_idx in range(start, end):
            k = int(ci_sorted[cand_idx].item())
            if k > j - 1:
                continue
            left = dp[:, : k + 1, k - 1] if k - 1 >= 0 else dp.new_zeros((comp_count, 1))
            inside = dp[:, k + 1, j - 1] if k + 1 <= j - 1 else dp.new_zeros((comp_count,))
            cand = left + comp_sorted[:, cand_idx].view(comp_count, 1) + inside.view(comp_count, 1)
            best_slice = best[:, : k + 1]
            better = cand[0] > best_slice[0]
            equal = cand[0] == best_slice[0]
            for c in range(1, comp_count):
                better |= equal & (cand[c] > best_slice[c])
                equal &= cand[c] == best_slice[c]
            better |= equal & (cand_idx > best_idx[: k + 1])
            if bool(better.any().item()):
                best[:, : k + 1][:, better] = cand[:, better]
                best_idx[: k + 1][better] = cand_idx
        dp[:, :j, j] = best
        choice[:j, j] = best_idx

    selected: List[int] = []
    stack = [(0, comp_len - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        idx = int(choice[i, j].item())
        if idx == -1:
            stack.append((i, j - 1))
            continue
        k = int(ci_sorted[idx].item())
        selected.append(int(order[idx].item()))
        if i <= k - 1:
            stack.append((i, k - 1))
        if k + 1 <= j - 1:
            stack.append((k + 1, j - 1))
    if not selected:
        return ci_sorted.new_empty((0,), dtype=torch.long)
    return torch.tensor(sorted(selected), dtype=torch.long, device=device)


def _numpy_split_pseudoknot_pairs(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pairs.size == 0:
        empty = pairs.reshape(0, 2)
        return empty, empty
    if pairs.shape[0] < 2:
        return pairs, pairs[:0]
    if not _numpy_crossing_mask(pairs).any():
        return pairs, pairs[:0]
    empty = pairs[:0]
    pair_i, pair_j, seg_start, seg_len = _numpy_pairs_to_duplex_segment_arrays(pairs)
    if seg_len.size == 0:
        return pairs, empty
    start_i = pair_i[seg_start]
    start_j = pair_j[seg_start]
    nested_idx = _numpy_mwis_segments(start_i, start_j, seg_len)
    if nested_idx.size == 0:
        return empty, pairs
    seg_mask = np.zeros_like(seg_len, dtype=bool)
    seg_mask[nested_idx] = True
    primary = segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=seg_mask, empty=empty)
    pseudoknot = segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=~seg_mask, empty=empty)
    return primary, pseudoknot


def _numpy_mwis_segments(start_i: np.ndarray, start_j: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    if start_i.size == 0:
        return np.empty((0,), dtype=int)
    _, ci, cj, comp_len = _numpy_compress_endpoints(start_i, start_j)
    if comp_len <= 1:
        return np.empty((0,), dtype=int)

    span = (start_j - start_i).astype(int, copy=False)
    unpaired = span + 1 - 2 * lengths
    comp_vals = np.stack([lengths, -unpaired, -span, -np.ones_like(lengths)], axis=0)
    seg_count = lengths.size
    seg_ids = np.arange(seg_count, dtype=int)
    order = np.argsort(cj * (seg_count + 1) + seg_ids)
    ci_sorted = ci[order]
    cj_sorted = cj[order]
    comp_sorted = comp_vals[:, order]
    _, prefix = _numpy_endpoint_prefix(cj_sorted, comp_len)

    return _numpy_mwis_select(ci_sorted, order, prefix, comp_sorted, comp_len)


def _numpy_mwis_select(
    ci_sorted: np.ndarray,
    order: np.ndarray,
    prefix: np.ndarray,
    comp_sorted: np.ndarray,
    comp_len: int,
) -> np.ndarray:
    comp_count = comp_sorted.shape[0]
    dp = np.zeros((comp_count, comp_len, comp_len), dtype=np.int64)
    choice = np.full((comp_len, comp_len), -1, dtype=np.int64)
    for j in range(1, comp_len):
        best = dp[:, :j, j - 1].copy()
        best_idx = np.full(j, -1, dtype=np.int64)
        start = int(prefix[j - 1]) if j > 0 else 0
        end = int(prefix[j])
        for cand_idx in range(start, end):
            k = int(ci_sorted[cand_idx])
            if k > j - 1:
                continue
            left = dp[:, : k + 1, k - 1] if k - 1 >= 0 else np.zeros((comp_count, 1), dtype=dp.dtype)
            inside = dp[:, k + 1, j - 1] if k + 1 <= j - 1 else np.zeros((comp_count,), dtype=dp.dtype)
            comp_vals = comp_sorted[:, cand_idx][:, None]
            cand = left + comp_vals + inside[:, None]
            best_slice = best[:, : k + 1]
            better = cand[0] > best_slice[0]
            equal = cand[0] == best_slice[0]
            for c in range(1, comp_count):
                better |= equal & (cand[c] > best_slice[c])
                equal &= cand[c] == best_slice[c]
            better |= equal & (cand_idx > best_idx[: k + 1])
            if np.any(better):
                best[:, : k + 1][:, better] = cand[:, better]
                best_idx[: k + 1][better] = cand_idx
        dp[:, :j, j] = best
        choice[:j, j] = best_idx

    selected: List[int] = []
    stack = [(0, comp_len - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        idx = int(choice[i, j])
        if idx == -1:
            stack.append((i, j - 1))
            continue
        k = int(ci_sorted[idx])
        selected.append(int(order[idx]))
        if i <= k - 1:
            stack.append((i, k - 1))
        if k + 1 <= j - 1:
            stack.append((k + 1, j - 1))
    if not selected:
        return np.empty((0,), dtype=int)
    return np.sort(np.array(selected, dtype=int))


def nested_pairs(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | PairsList:
    """
    Return primary pairs from the segment-MWIS split.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Primary pairs using the same backend as input.

    This is equivalent to ``split_pseudoknot_pairs(pairs)[0]`` and expects
    normalized unique pairs.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> nested_pairs(torch.tensor([[0, 2], [1, 3]])).tolist()
        [[1, 3]]

        NumPy input
        >>> import numpy as np
        >>> nested_pairs(np.array([[0, 2], [1, 3]])).tolist()
        [[1, 3]]

        List input
        >>> nested_pairs([(0, 2), (1, 3)])
        [(1, 3)]
    """
    primary, _ = split_pseudoknot_pairs(pairs)
    return primary


@overload
def pseudoknot_pairs(pairs: np.ndarray) -> np.ndarray: ...


@overload
def pseudoknot_pairs(pairs: Pairs) -> PairsList: ...  # type: ignore[overload-cannot-match]


@overload
def pseudoknot_pairs(pairs: Tensor) -> Tensor: ...  # type: ignore[overload-cannot-match]


def pseudoknot_pairs(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | PairsList:
    """
    Return pseudoknot pairs from segments not selected by MWIS.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Pseudoknot pairs using the same backend as input.

    This is equivalent to ``split_pseudoknot_pairs(pairs)[1]`` and expects
    normalized unique pairs.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Tie-breaks for equal total base pairs: (1) minimize unpaired-within-span,
    (2) minimize total span, (3) minimize number of segments, (4) deterministic
    order fallback.

    Examples:
        Torch input
        >>> import torch
        >>> pseudoknot_pairs(torch.tensor([[0, 2], [1, 3]])).tolist()
        [[0, 2]]

        NumPy input
        >>> import numpy as np
        >>> pseudoknot_pairs(np.array([[0, 2], [1, 3]])).tolist()
        [[0, 2]]

        List input
        >>> pseudoknot_pairs([(0, 2), (1, 3)])
        [(0, 2)]
    """
    _, pseudoknot = split_pseudoknot_pairs(pairs)
    return pseudoknot


def pseudoknot_nucleotides(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | List[int]:
    """
    Return nucleotide indices involved in any pseudoknot pair.

    Pair inputs are expected to be normalized.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Unique nucleotide indices using the same backend as input (sequence inputs return Python lists).

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> pseudoknot_nucleotides(torch.tensor([[0, 2], [1, 3]])).tolist()
        [0, 2]

        NumPy input
        >>> import numpy as np
        >>> pseudoknot_nucleotides(np.array([[0, 2], [1, 3]])).tolist()
        [0, 2]

        List input
        >>> pseudoknot_nucleotides([(0, 2), (1, 3)])
        [0, 2]
    """
    if isinstance(pairs, Tensor):
        pseudoknot_pairs_data = pseudoknot_pairs(pairs)
        if pseudoknot_pairs_data.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=pairs.device)
        return torch.unique(pseudoknot_pairs_data.view(-1)).to(torch.long)
    if isinstance(pairs, np.ndarray):
        pseudoknot_pairs_data = pseudoknot_pairs(pairs)
        if pseudoknot_pairs_data.size == 0:
            return np.empty((0,), dtype=int)
        return np.unique(pseudoknot_pairs_data.reshape(-1))
    if isinstance(pairs, Sequence):
        pseudoknot_pairs_data = pseudoknot_pairs(pairs)
        if not pseudoknot_pairs_data:
            return []
        return np.unique(np.asarray(pseudoknot_pairs_data, dtype=int).reshape(-1)).tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def has_pseudoknot(pairs: Tensor | np.ndarray | Pairs) -> bool:
    """
    Return True if any pseudoknot pairs are present under segment-MWIS split.

    Pair inputs are expected to be normalized.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        True if pseudoknot pairs exist, otherwise False.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> has_pseudoknot(torch.tensor([[0, 2], [1, 3]]))
        True

        NumPy input
        >>> import numpy as np
        >>> has_pseudoknot(np.array([[0, 2], [1, 3]]))
        True

        List input
        >>> has_pseudoknot([(0, 2), (1, 3)])
        True
        >>> has_pseudoknot([(0, 3), (1, 2)])
        False
    """
    _, pseudoknot_pairs = split_pseudoknot_pairs(pairs)
    if isinstance(pseudoknot_pairs, Tensor):
        return bool(pseudoknot_pairs.numel())
    if isinstance(pseudoknot_pairs, np.ndarray):
        return bool(pseudoknot_pairs.size)
    if isinstance(pseudoknot_pairs, Sequence):
        return bool(pseudoknot_pairs)
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


@overload
def crossing_events(pairs: Tensor) -> Tensor: ...


@overload
def crossing_events(pairs: np.ndarray) -> np.ndarray: ...  # type: ignore[overload-cannot-match]


@overload
def crossing_events(  # type: ignore[overload-cannot-match]
    pairs: Pairs,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]: ...


def crossing_events(
    pairs: Tensor | np.ndarray | Pairs,
) -> Tensor | np.ndarray | List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """
    Return crossing events between stem segments.

    Each stem is encoded as (start_5p, stop_5p, start_3p, stop_3p).

    For pair-level crossing arcs, use ``crossing_arcs``.

    Pair inputs are expected to be normalized.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Crossing events using the same backend as input. Tensor/NumPy inputs
        return shape (n, 2, 4).

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> crossing_events(torch.tensor([[0, 2], [1, 3]])).tolist()
        [[[0, 0, 2, 2], [1, 1, 3, 3]]]

        NumPy input
        >>> import numpy as np
        >>> crossing_events(np.array([[0, 2], [1, 3]])).tolist()
        [[[0, 0, 2, 2], [1, 1, 3, 3]]]

        List input
        >>> crossing_events([(0, 2), (1, 3)])
        [((0, 0, 2, 2), (1, 1, 3, 3))]
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_crossing_events(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_crossing_events(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        events = _numpy_crossing_events(pairs)
        return [(tuple(stem_a), tuple(stem_b)) for stem_a, stem_b in events.tolist()]
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_crossing_events(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0:
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    if pairs.shape[0] < 2:
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    start_i, start_j, lengths = _torch_pairs_to_stem_segment_arrays(pairs)
    if start_i.numel() == 0:
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    start_i = start_i.to(dtype=torch.long)
    start_j = start_j.to(dtype=torch.long)
    lengths = lengths.to(dtype=torch.long)
    stop_i = start_i + lengths - 1
    stop_j = start_j - lengths + 1
    stems = torch.stack([start_i, stop_i, start_j, stop_j], dim=1)
    if stems.shape[0] < 2:
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    valid = stems[:, 0] < stems[:, 2]
    if not bool(valid.any().item()):
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    stems = stems[valid]
    if stems.shape[0] < 2:
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    base = int(stems[:, 2].max().item()) + 1
    order = torch.argsort(stems[:, 0] * base + stems[:, 2])
    stems = stems[order]

    events: List[Tensor] = []
    count = int(stems.shape[0])
    for idx in range(count):
        a5 = stems[idx, 0]
        a3 = stems[idx, 2]
        for jdx in range(idx + 1, count):
            b5 = stems[jdx, 0]
            if b5 >= a3:
                break
            b3 = stems[jdx, 2]
            if a5 < b5 < a3 < b3:
                events.append(torch.stack([stems[idx], stems[jdx]], dim=0))
    if not events:
        return pairs.new_empty((0, 2, 4), dtype=torch.long)
    return torch.stack(events, dim=0)


def _numpy_crossing_events(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0, 2, 4), dtype=int)
    if pairs.shape[0] < 2:
        return np.empty((0, 2, 4), dtype=int)
    start_i, start_j, lengths = _numpy_pairs_to_stem_segment_arrays(pairs)
    if len(start_i) == 0:
        return np.empty((0, 2, 4), dtype=int)
    start_i = np.asarray(start_i, dtype=int)
    start_j = np.asarray(start_j, dtype=int)
    lengths = np.asarray(lengths, dtype=int)
    stop_i = start_i + lengths - 1
    stop_j = start_j - lengths + 1
    stems = np.stack([start_i, stop_i, start_j, stop_j], axis=1)
    if stems.shape[0] < 2:
        return np.empty((0, 2, 4), dtype=int)
    valid = stems[:, 0] < stems[:, 2]
    if not np.any(valid):
        return np.empty((0, 2, 4), dtype=int)
    stems = stems[valid]
    if stems.shape[0] < 2:
        return np.empty((0, 2, 4), dtype=int)
    base = int(stems[:, 2].max()) + 1
    order = np.argsort(stems[:, 0] * base + stems[:, 2])
    stems = stems[order]

    events: List[np.ndarray] = []
    count = int(stems.shape[0])
    for idx in range(count):
        a5 = stems[idx, 0]
        a3 = stems[idx, 2]
        for jdx in range(idx + 1, count):
            b5 = stems[jdx, 0]
            if b5 >= a3:
                break
            b3 = stems[jdx, 2]
            if a5 < b5 < a3 < b3:
                events.append(np.stack([stems[idx], stems[jdx]], axis=0))
    if not events:
        return np.empty((0, 2, 4), dtype=int)
    return np.stack(events, axis=0).astype(int, copy=False)


@overload
def crossing_arcs(pairs: Tensor) -> Tensor: ...


@overload
def crossing_arcs(pairs: np.ndarray) -> np.ndarray: ...  # type: ignore[overload-cannot-match]


@overload
def crossing_arcs(pairs: Pairs) -> List[Tuple[Pair, Pair]]: ...  # type: ignore[overload-cannot-match]


def crossing_arcs(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | List[Tuple[Pair, Pair]]:
    """
    Return pair-level crossing arcs as ((i, j), (k, l)) where i < k < j < l.

    Pair inputs are expected to be normalized.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Crossing arcs using the same backend as input. Tensor/NumPy inputs
        return shape (n, 2, 2).

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> crossing_arcs(torch.tensor([[0, 2], [1, 3]])).tolist()
        [[[0, 2], [1, 3]]]

        NumPy input
        >>> import numpy as np
        >>> crossing_arcs(np.array([[0, 2], [1, 3]])).tolist()
        [[[0, 2], [1, 3]]]

        List input
        >>> crossing_arcs([(0, 2), (1, 3)])
        [((0, 2), (1, 3))]
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_crossing_arcs(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_crossing_arcs(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        events = _numpy_crossing_arcs(pairs)
        return [(tuple(pair_a), tuple(pair_b)) for pair_a, pair_b in events.tolist()]
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_crossing_arcs(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0:
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    if pairs.shape[0] < 2:
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    pairs = torch.unique(pairs.to(torch.long), dim=0)
    if pairs.shape[0] < 2:
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    base = int(pairs.max().item()) + 1
    key = pairs[:, 0] * base + pairs[:, 1]
    pairs = pairs[torch.argsort(key)]
    i = pairs[:, 0]
    j = pairs[:, 1]
    ii = i.view(-1, 1)
    jj = j.view(-1, 1)
    kk = i.view(1, -1)
    ll = j.view(1, -1)
    cond = (ii < kk) & (kk < jj) & (jj < ll)
    cond = torch.triu(cond, diagonal=1)
    if not bool(cond.any().item()):
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    a_idx, b_idx = torch.where(cond)
    if a_idx.numel() == 0:
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    pair_a = torch.stack([i[a_idx], j[a_idx]], dim=1)
    pair_b = torch.stack([i[b_idx], j[b_idx]], dim=1)
    events = torch.stack([pair_a, pair_b], dim=1).to(torch.long)
    flat = events.reshape(-1, 4)
    if flat.numel() == 0:
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    unique = torch.unique(flat, dim=0)
    if unique.numel() == 0:
        return pairs.new_empty((0, 2, 2), dtype=torch.long)
    return unique.reshape(-1, 2, 2)


def _numpy_crossing_arcs(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0, 2, 2), dtype=int)
    if pairs.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=int)
    pairs = np.unique(pairs.astype(int, copy=False), axis=0)
    if pairs.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=int)
    base = int(pairs.max()) + 1
    key = pairs[:, 0] * base + pairs[:, 1]
    pairs = pairs[np.argsort(key)]
    i = pairs[:, 0]
    j = pairs[:, 1]
    ii = i[:, None]
    jj = j[:, None]
    kk = i[None, :]
    ll = j[None, :]
    cond = (ii < kk) & (kk < jj) & (jj < ll)
    cond = np.triu(cond, k=1)
    if not np.any(cond):
        return np.empty((0, 2, 2), dtype=int)
    a_idx, b_idx = np.where(cond)
    if a_idx.size == 0:
        return np.empty((0, 2, 2), dtype=int)
    pair_a = np.stack([i[a_idx], j[a_idx]], axis=1)
    pair_b = np.stack([i[b_idx], j[b_idx]], axis=1)
    events = np.stack([pair_a, pair_b], axis=1)
    if events.size == 0:
        return np.empty((0, 2, 2), dtype=int)
    flat = events.reshape(-1, 4)
    unique = np.unique(flat, axis=0)
    if unique.size == 0:
        return np.empty((0, 2, 2), dtype=int)
    return unique.reshape(-1, 2, 2)


@overload
def split_crossing_pairs(pairs: Tensor) -> Tuple[Tensor, Tensor]: ...


@overload
def split_crossing_pairs(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...  # type: ignore[overload-cannot-match]


@overload
def split_crossing_pairs(pairs: PairsList) -> Tuple[PairsList, PairsList]: ...  # type: ignore[overload-cannot-match]


def split_crossing_pairs(
    pairs: Tensor | np.ndarray | Pairs,
) -> Tuple[Tensor | np.ndarray | PairsList, Tensor | np.ndarray | PairsList]:
    """
    Split pairs into non-crossing pairs and crossing pairs (no-heuristic).

    Pairs are expected to be normalized (unique, sorted with i < j).
    Use ``normalize_pairs`` if you need to normalize raw inputs.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        (non_crossing_pairs, crossing_pairs) using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> primary, crossing = split_crossing_pairs(torch.tensor([[0, 2], [1, 3]]))
        >>> primary.tolist(), crossing.tolist()
        ([], [[0, 2], [1, 3]])

        NumPy input
        >>> import numpy as np
        >>> primary, crossing = split_crossing_pairs(np.array([[0, 3], [1, 2]]))
        >>> primary.tolist(), crossing.tolist()
        ([[0, 3], [1, 2]], [])

        List input
        >>> split_crossing_pairs([(0, 2), (1, 3)])
        ([], [(0, 2), (1, 3)])
        >>> split_crossing_pairs([(0, 3), (1, 2)])
        ([(0, 3), (1, 2)], [])
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            empty = pairs.view(0, 2)
            return empty, empty
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        if pairs.shape[0] < 2:
            return _torch_sort_pairs(pairs), pairs.new_empty((0, 2))
        mask = _torch_crossing_mask(pairs)
        if not bool(mask.any().item()):
            return _torch_sort_pairs(pairs), pairs.new_empty((0, 2))
        primary = _torch_sort_pairs(pairs[~mask])
        crossing = _torch_sort_pairs(pairs[mask])
        return primary, crossing
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            empty = pairs.reshape(0, 2)
            return empty, empty
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        if pairs.shape[0] < 2:
            return _numpy_sort_pairs(pairs), pairs[:0]
        mask = _numpy_crossing_mask(pairs)
        if not mask.any():
            return _numpy_sort_pairs(pairs), pairs[:0]
        primary = _numpy_sort_pairs(pairs[~mask])
        crossing = _numpy_sort_pairs(pairs[mask])
        return primary, crossing
    if isinstance(pairs, Sequence):
        if not pairs:
            return [], []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        mask = _numpy_crossing_mask(pairs)
        if not mask.any():
            return list(map(tuple, _numpy_sort_pairs(pairs).tolist())), []
        primary = _numpy_sort_pairs(pairs[~mask]).tolist()
        crossing = _numpy_sort_pairs(pairs[mask]).tolist()
        return list(map(tuple, primary)), list(map(tuple, crossing))
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


@overload
def crossing_pairs(pairs: np.ndarray) -> np.ndarray: ...


@overload
def crossing_pairs(pairs: Pairs) -> PairsList: ...  # type: ignore[overload-cannot-match]


@overload
def crossing_pairs(pairs: Tensor) -> Tensor: ...  # type: ignore[overload-cannot-match]


def crossing_pairs(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | PairsList:
    """
    Return pairs from segments that cross any other segment (no-heuristic PK).

    Pairs are expected to be normalized (unique, sorted with i < j).
    Use ``normalize_pairs`` if you need to normalize raw inputs.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Crossing pairs using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> crossing_pairs(torch.tensor([[0, 2], [1, 3]])).tolist()
        [[0, 2], [1, 3]]

        NumPy input
        >>> import numpy as np
        >>> crossing_pairs(np.array([[0, 2], [1, 3]])).tolist()
        [[0, 2], [1, 3]]

        List input
        >>> crossing_pairs([(0, 2), (1, 3)])
        [(0, 2), (1, 3)]
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_crossing_pairs(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_crossing_pairs(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return list(map(tuple, _numpy_crossing_pairs(pairs).tolist()))
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_crossing_pairs(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0:
        return pairs.view(0, 2)
    if pairs.shape[0] < 2:
        return pairs.new_empty((0, 2))
    if not bool(_torch_crossing_mask(pairs).any().item()):
        return pairs.new_empty((0, 2))
    pair_i, pair_j, seg_start, seg_len = _torch_pairs_to_duplex_segment_arrays(pairs)
    if seg_len.numel() < 2:
        return pairs.new_empty((0, 2))
    outer_pairs = torch.stack([pair_i[seg_start], pair_j[seg_start]], dim=1)
    if outer_pairs.shape[0] < 2:
        return pairs.new_empty((0, 2))
    seg_mask = _torch_crossing_mask(outer_pairs)
    if not bool(seg_mask.any().item()):
        return pairs.new_empty((0, 2))
    crossing_pairs = segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=seg_mask)
    if crossing_pairs.numel() == 0:
        return crossing_pairs
    return _torch_sort_pairs(crossing_pairs)


def _numpy_crossing_pairs(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    empty = pairs[:0]
    if pairs.shape[0] < 2:
        return empty
    pair_i, pair_j, seg_start, seg_len = _numpy_pairs_to_duplex_segment_arrays(pairs)
    if seg_len.size < 2:
        return empty
    outer_pairs = np.stack((pair_i[seg_start], pair_j[seg_start]), axis=1)
    if outer_pairs.size == 0:
        return empty
    seg_mask = _numpy_crossing_mask(outer_pairs)
    if not np.any(seg_mask):
        return empty
    return segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=seg_mask, empty=empty)


def crossing_nucleotides(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | List[int]:
    """
    Return nucleotide indices involved in any crossing pair.

    Pair inputs are expected to be normalized.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Unique nucleotide indices using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> crossing_nucleotides(torch.tensor([[0, 2], [1, 3]])).tolist()
        [0, 1, 2, 3]

        NumPy input
        >>> import numpy as np
        >>> crossing_nucleotides(np.array([[0, 2], [1, 3]])).tolist()
        [0, 1, 2, 3]

        List input
        >>> crossing_nucleotides([(0, 2), (1, 3)])
        [0, 1, 2, 3]
    """
    if isinstance(pairs, Tensor):
        crossing = crossing_pairs(pairs)
        if crossing.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=pairs.device)
        return torch.unique(crossing.view(-1)).to(torch.long)
    if isinstance(pairs, np.ndarray):
        crossing = crossing_pairs(pairs)
        if crossing.size == 0:
            return np.empty((0,), dtype=int)
        return np.unique(crossing.reshape(-1))
    if isinstance(pairs, Sequence):
        crossing = crossing_pairs(pairs)
        if not crossing:
            return []
        return np.unique(np.asarray(crossing, dtype=int).reshape(-1)).tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def crossing_mask(pairs: Tensor | np.ndarray | Pairs) -> Tensor | np.ndarray | List[bool]:
    """
    Return a boolean mask for pairs that cross any other pair.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.

    Returns:
        Boolean mask for the input pairs using the same backend as input.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> crossing_mask(torch.tensor([[0, 2], [1, 3]])).tolist()
        [True, True]

        NumPy input
        >>> import numpy as np
        >>> crossing_mask(np.array([[0, 2], [1, 3]])).tolist()
        [True, True]

        List input
        >>> crossing_mask([(0, 2), (1, 3)])
        [True, True]
        >>> crossing_mask([(0, 3), (1, 2)])
        [False, False]
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_crossing_mask(pairs)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_crossing_mask(pairs)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return _numpy_crossing_mask(pairs).tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_crossing_mask(norm_pairs: Tensor) -> Tensor:
    if norm_pairs.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=norm_pairs.device)
    n = norm_pairs.shape[0]
    if n < 2:
        return torch.zeros((n,), dtype=torch.bool, device=norm_pairs.device)
    pairs = norm_pairs.to(torch.long)

    ii = pairs[:, 0]
    jj = pairs[:, 1]
    max_j = int(jj.max().item()) if jj.numel() else 0
    key = ii * (max_j + 1) + jj
    order = torch.argsort(key)
    pairs = pairs[order]
    ii = pairs[:, 0]
    jj = pairs[:, 1]

    max_pos = int(torch.maximum(ii.max(), jj.max()).item())
    length = max_pos + 1
    device = norm_pairs.device
    start_to_end = torch.full((length,), -1, dtype=torch.long, device=device)
    if hasattr(start_to_end, "scatter_reduce_"):
        start_to_end.scatter_reduce_(0, ii, jj, reduce="amax", include_self=True)
    else:
        for idx in range(ii.numel()):
            i = int(ii[idx].item())
            j = int(jj[idx].item())
            if j > int(start_to_end[i].item()):
                start_to_end[i] = j

    st = [start_to_end]
    step = 1
    while step < length:
        prev = st[-1]
        if prev.numel() <= step:
            break
        st.append(torch.maximum(prev[:-step], prev[step:]))
        step <<= 1

    levels = len(st)
    st_padded = torch.full((levels, length), -1, dtype=torch.long, device=device)
    for level_idx, table in enumerate(st):
        st_padded[level_idx, : table.numel()] = table

    l = ii + 1  # noqa: E741
    r = jj - 1
    valid = l <= r
    if valid.any():
        span = (r - l + 1).clamp(min=1).to(torch.float32)
        k = torch.floor(torch.log2(span)).to(torch.long)
        pow2 = 1 << torch.arange(levels, dtype=torch.long, device=device)
        span_len = pow2[k]
        right_idx = r - span_len + 1
        left = st_padded[k, l]
        right = st_padded[k, right_idx]
        max_lr = torch.maximum(left, right)
        mask_a = valid & (max_lr > jj)
    else:
        mask_a = torch.zeros((n,), dtype=torch.bool, device=device)

    bit_size = length + 1
    bit = torch.zeros(bit_size + 1, dtype=torch.int64, device=device)

    def bit_add(pos: int) -> None:
        idx = pos + 1
        while idx <= bit_size:
            bit[idx] += 1
            idx += idx & -idx

    def bit_sum(pos: int) -> int:
        if pos < 0:
            return 0
        idx = pos + 1
        total = 0
        while idx > 0:
            total += int(bit[idx].item())
            idx -= idx & -idx
        return total

    mask_b = torch.zeros((n,), dtype=torch.bool, device=device)
    idx = 0
    while idx < n:
        start = int(ii[idx].item())
        end_idx = idx + 1
        while end_idx < n and int(ii[end_idx].item()) == start:
            end_idx += 1
        for pos in range(idx, end_idx):
            l_pos = start + 1
            r_pos = int(jj[pos].item()) - 1
            if l_pos <= r_pos and (bit_sum(r_pos) - bit_sum(l_pos - 1)) > 0:
                mask_b[pos] = True
        for pos in range(idx, end_idx):
            bit_add(int(jj[pos].item()))
        idx = end_idx

    mask = mask_a | mask_b
    mask_unsorted = torch.empty_like(mask)
    mask_unsorted[order] = mask
    return mask_unsorted


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


def pseudoknot_tiers(
    pairs: Tensor | np.ndarray | Pairs, unsafe: bool = False
) -> List[Tensor] | List[np.ndarray] | Tiers:
    """
    Return dot-bracket tiers as non-crossing groups of pairs.

    Pairs are expected to be normalized (unique, sorted with i < j).
    Use ``normalize_pairs`` if you need to normalize raw inputs.

    Args:
        pairs: torch.Tensor, numpy.ndarray, or array-like with shape (n, 2) and 0-based indices.
        unsafe: Use greedy tiering for speed instead of minimal coloring.

    Returns:
        A list of tiers. Each tier is a list/array/tensor of pairs.

    Raises:
        ValueError: If pairs has invalid shape for the selected backend.
        TypeError: If pairs is not a torch.Tensor, numpy.ndarray, or array-like with shape (n, 2).

    Examples:
        Torch input
        >>> import torch
        >>> tiers = pseudoknot_tiers(torch.tensor([[0, 2], [1, 3]]))
        >>> [tier.tolist() for tier in tiers]
        [[[0, 2]], [[1, 3]]]

        NumPy input
        >>> import numpy as np
        >>> tiers = pseudoknot_tiers(np.array([[0, 2], [1, 3]]))
        >>> [tier.tolist() for tier in tiers]
        [[[0, 2]], [[1, 3]]]

        List input
        >>> pseudoknot_tiers([(0, 3), (1, 2)])
        [[(0, 3), (1, 2)]]
    """
    if isinstance(pairs, Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_tiers_from_pairs(pairs, unsafe=unsafe)
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_tiers_from_pairs(pairs, unsafe=unsafe)
    if isinstance(pairs, Sequence):
        if not pairs:
            return []
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        tiers = _numpy_tiers_from_pairs(pairs, unsafe=unsafe)
        return [list(map(tuple, tier.tolist())) for tier in tiers]
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_tiers_from_pairs(pairs: Tensor, unsafe: bool = False) -> List[Tensor]:
    if pairs.numel() == 0:
        return []
    if not bool(_torch_crossing_mask(pairs).any().item()):
        return [pairs]
    if unsafe:
        return _torch_greedy_pseudoknot_tiers(pairs)
    tiers = _numpy_minimal_pseudoknot_tiers(pairs.detach().cpu().numpy())
    out: List[Tensor] = []
    for tier in tiers:
        if tier:
            out.append(torch.tensor(tier, dtype=torch.long, device=pairs.device))
        else:
            out.append(pairs.new_empty((0, 2), dtype=torch.long))
    return out


def _torch_greedy_pseudoknot_tiers(pairs: Tensor) -> List[Tensor]:
    tiers: Tiers = []
    end_stacks: List[List[int]] = []
    device = pairs.device
    for a, b in pairs.detach().cpu().tolist():
        placed = False
        for tier, stack in zip(tiers, end_stacks):
            while stack and stack[-1] < a:
                stack.pop()
            if not stack or b < stack[-1]:
                tier.append((a, b))
                stack.append(b)
                placed = True
                break
        if not placed:
            tiers.append([(a, b)])
            end_stacks.append([b])
    out: List[Tensor] = []
    for tier in tiers:
        if tier:
            out.append(torch.tensor(tier, dtype=torch.long, device=device))
        else:
            out.append(pairs.new_empty((0, 2), dtype=torch.long))
    return out


def _numpy_tiers_from_pairs(pairs: np.ndarray, unsafe: bool = False) -> List[np.ndarray]:
    if pairs.size == 0:
        return []
    if not _numpy_crossing_mask(pairs).any():
        return [pairs]
    tiers = _numpy_greedy_pseudoknot_tiers(pairs) if unsafe else _numpy_minimal_pseudoknot_tiers(pairs)
    out: List[np.ndarray] = []
    for tier in tiers:
        if tier:
            out.append(np.array(tier, dtype=int))
        else:
            out.append(np.empty((0, 2), dtype=int))
    return out


def _numpy_greedy_pseudoknot_tiers(pairs: np.ndarray) -> Tiers:
    tiers: Tiers = []
    end_stacks: List[List[int]] = []
    for a, b in pairs.tolist():
        placed = False
        for tier_k, stack in zip(tiers, end_stacks):
            while stack and stack[-1] < a:
                stack.pop()
            if not stack or b < stack[-1]:
                tier_k.append((a, b))
                stack.append(b)
                placed = True
                break
        if not placed:
            tiers.append([(a, b)])
            end_stacks.append([b])
    return tiers


def _numpy_minimal_pseudoknot_tiers(pairs: np.ndarray) -> Tiers:
    if pairs.size == 0:
        return []
    adj = _numpy_crossing_adjacency(pairs)
    colors, num_colors = _numpy_dsatur_min_coloring(adj)
    tiers: Tiers = [[] for _ in range(num_colors)]
    for (a, b), color in zip(pairs.tolist(), colors):
        tiers[color].append((a, b))
    return tiers


def _numpy_crossing_adjacency(pairs: np.ndarray) -> List[List[int]]:
    n = pairs.shape[0]
    adj: List[List[int]] = [[] for _ in range(n)]
    ii = pairs[:, 0]
    jj = pairs[:, 1]
    for i in range(n - 1):
        ai = int(ii[i])
        aj = int(jj[i])
        for j in range(i + 1, n):
            bi = int(ii[j])
            bj = int(jj[j])
            if _pairs_cross(ai, aj, bi, bj):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _pairs_cross(a_i: int, a_j: int, b_i: int, b_j: int) -> bool:
    return (a_i < b_i < a_j < b_j) or (b_i < a_i < b_j < a_j)


def _numpy_dsatur_min_coloring(adj: List[List[int]]) -> Tuple[List[int], int]:
    n = len(adj)
    if n == 0:
        return [], 0
    degrees = [len(neighbors) for neighbors in adj]
    best_colors, best = _numpy_dsatur_greedy_coloring(adj, degrees)
    if best <= 1:
        return best_colors, best

    colors = [-1] * n
    neighbor_color_counts: List[Dict[int, int]] = [{} for _ in range(n)]
    uncolored = [True] * n
    remaining = n

    def assign(node: int, color: int) -> None:
        nonlocal remaining
        colors[node] = color
        uncolored[node] = False
        remaining -= 1
        for neighbor in adj[node]:
            counts = neighbor_color_counts[neighbor]
            counts[color] = counts.get(color, 0) + 1

    def unassign(node: int, color: int) -> None:
        nonlocal remaining
        colors[node] = -1
        uncolored[node] = True
        remaining += 1
        for neighbor in adj[node]:
            counts = neighbor_color_counts[neighbor]
            new_count = counts[color] - 1
            if new_count == 0:
                del counts[color]
            else:
                counts[color] = new_count

    def backtrack(num_colors: int) -> None:
        nonlocal best, best_colors
        if remaining == 0:
            if num_colors < best:
                best = num_colors
                best_colors = colors.copy()
            return
        if num_colors >= best:
            return
        node = _numpy_select_dsatur_vertex(uncolored, neighbor_color_counts, degrees)
        used = neighbor_color_counts[node]

        for color in range(num_colors):
            if color in used:
                continue
            assign(node, color)
            backtrack(num_colors)
            unassign(node, color)

        if num_colors + 1 < best:
            color = num_colors
            assign(node, color)
            backtrack(num_colors + 1)
            unassign(node, color)

    backtrack(0)
    return best_colors, best


def _numpy_dsatur_greedy_coloring(adj: List[List[int]], degrees: List[int]) -> Tuple[List[int], int]:
    n = len(adj)
    colors = [-1] * n
    neighbor_color_counts: List[Dict[int, int]] = [{} for _ in range(n)]
    uncolored = [True] * n
    remaining = n
    num_colors = 0

    while remaining:
        v = _numpy_select_dsatur_vertex(uncolored, neighbor_color_counts, degrees)
        used = neighbor_color_counts[v]
        color = 0
        while color in used:
            color += 1
        colors[v] = color
        if color == num_colors:
            num_colors += 1
        uncolored[v] = False
        remaining -= 1
        for u in adj[v]:
            counts = neighbor_color_counts[u]
            counts[color] = counts.get(color, 0) + 1

    return colors, num_colors


def _numpy_select_dsatur_vertex(
    uncolored: List[bool], neighbor_color_counts: List[Dict[int, int]], degrees: List[int]
) -> int:
    best_idx = -1
    best_key = (-1, -1, 0)
    for idx, is_uncolored in enumerate(uncolored):
        if not is_uncolored:
            continue
        key = (len(neighbor_color_counts[idx]), degrees[idx], -idx)
        if key > best_key:
            best_key = key
            best_idx = idx
    return best_idx
