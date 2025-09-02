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

import string
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Dict, List, overload
from warnings import warn

import numpy as np
import torch
from torch import Tensor

from .pairs import Pairs, PairsList, _numpy_normalize_pairs_low_high, _torch_normalize_pairs_low_high
from .pseudoknot import pseudoknot_tiers

_DOT_BRACKET_PAIR_TABLE: Dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
_DOT_BRACKET_PAIR_TABLE.update(zip(string.ascii_uppercase, string.ascii_lowercase))
_REVERSE_DOT_BRACKET_PAIR_TABLE: Mapping[str, str] = {v: k for k, v in _DOT_BRACKET_PAIR_TABLE.items()}
_UNPAIRED_TOKENS = {"+", ".", ",", "_"}


def dot_bracket_to_pairs(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a list of base-pair indices.

    Args:
        dot_bracket: Dot-bracket notation. Supports pseudoknots via multiple
            bracket types, including (), [], {}, <>, and A-Z/a-z. Unpaired
            tokens (`.`, `+`, `_`, `,`) are treated as unpaired positions.

    Returns:
        A numpy array of shape (n, 2) with pairs ``(i, j)`` where ``0 <= i < j < len(dot_bracket)``.

    Raises:
        ValueError: On unmatched or invalid symbols.

    Examples:
        >>> dot_bracket_to_pairs("((.))").tolist()
        [[0, 4], [1, 3]]
        >>> dot_bracket_to_pairs("([)]").tolist()
        [[0, 2], [1, 3]]
        >>> dot_bracket_to_pairs("...").tolist()
        []
    """
    stacks: defaultdict[str, List[int]] = defaultdict(list)
    pairs: PairsList = []
    for i, symbol in enumerate(dot_bracket):
        if symbol in _DOT_BRACKET_PAIR_TABLE:
            stacks[symbol].append(i)
        elif symbol in _REVERSE_DOT_BRACKET_PAIR_TABLE:
            opener = _REVERSE_DOT_BRACKET_PAIR_TABLE[symbol]
            try:
                j = stacks[opener].pop()
            except IndexError:
                raise ValueError(f"Unmatched symbol {symbol} at position {i} in sequence {dot_bracket}") from None
            pairs.append((j, i))
        elif symbol not in _UNPAIRED_TOKENS:
            raise ValueError(f"Invalid symbol {symbol} at position {i} in sequence {dot_bracket}")
    for symbol, stack in stacks.items():
        if stack:
            raise ValueError(f"Unmatched symbol {symbol} at position {stack[0]} in sequence {dot_bracket}")
    if not pairs:
        return np.empty((0, 2), dtype=int)
    pairs.sort()
    return np.asarray(pairs, dtype=int)


@overload
def pairs_to_contact_map(pairs: Tensor, length: int | None = None, unsafe: bool = False) -> Tensor: ...


@overload
def pairs_to_contact_map(  # type: ignore[overload-cannot-match]
    pairs: np.ndarray, length: int | None = None, unsafe: bool = False
) -> np.ndarray: ...


@overload
def pairs_to_contact_map(  # type: ignore[overload-cannot-match]
    pairs: PairsList, length: int | None = None, unsafe: bool = False
) -> List[List[bool]]: ...


def pairs_to_contact_map(
    pairs: Tensor | np.ndarray | Pairs,
    length: int | None = None,
    unsafe: bool = False,
) -> Tensor | np.ndarray | List[List[bool]]:
    """
    Convert base pairs to a symmetric contact map.

    If ``pairs`` is a torch tensor, returns a boolean torch.Tensor on the same device.
    Otherwise, returns a numpy boolean array.
    If ``length`` is None, it is inferred as ``max(pairs) + 1``.

    Examples:
        Torch input
        >>> import torch
        >>> contact_map_tensor = pairs_to_contact_map(torch.tensor([[0, 3], [1, 2]]), length=4)
        >>> contact_map_tensor.to(torch.int).tolist()
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]

        NumPy input
        >>> import numpy as np
        >>> contact_map = pairs_to_contact_map(np.array([(0, 3), (1, 2)]), length=4)
        >>> contact_map.astype(int).tolist()
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        >>> pairs_to_contact_map(np.array([(0, 2)])).astype(int).tolist()
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]]

        List input
        >>> pairs_to_contact_map([(0, 2)])
        [[False, False, True], [False, False, False], [True, False, False]]
    """
    if isinstance(pairs, Tensor):
        if pairs.numel() == 0:
            return _torch_pairs_to_contact_map(pairs.view(0, 2), length, unsafe)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a torch.Tensor with shape (n, 2)")
        return _torch_pairs_to_contact_map(pairs, length, unsafe)
    if isinstance(pairs, np.ndarray):
        if pairs.size == 0:
            return _numpy_pairs_to_contact_map(pairs.reshape(0, 2), length, unsafe)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a numpy.ndarray with shape (n, 2)")
        return _numpy_pairs_to_contact_map(pairs, length, unsafe)
    if isinstance(pairs, Sequence):
        pairs = np.asarray(pairs, dtype=int)
        if pairs.size == 0:
            return _numpy_pairs_to_contact_map(pairs.reshape(0, 2), length, unsafe)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be an array-like with shape (n, 2)")
        return _numpy_pairs_to_contact_map(pairs, length, unsafe).tolist()
    raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")


def _torch_pairs_to_contact_map(pairs: Tensor, length: int | None, unsafe: bool) -> Tensor:
    if pairs.numel() == 0:
        max_index = -1
        device = pairs.device
    else:
        pairs = _torch_normalize_pairs_low_high(pairs)
        max_index = int(torch.max(pairs).item())
        device = pairs.device

    if length is None:
        length = max_index + 1 if max_index >= 0 else 0
    contact_map_t = torch.zeros((length, length), dtype=torch.bool, device=device)
    if pairs.numel() == 0:
        return contact_map_t

    if torch.any(pairs[:, 0] == pairs[:, 1]):
        if not unsafe:
            raise ValueError("Self-pairing (i == j) is not allowed in pairs.")
        warn("Self-pairing (i == j) is not allowed in pairs.\nIgnoring such pairs.")
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return contact_map_t

    if torch.any((pairs < 0) | (pairs >= length)):
        mask_row = ((pairs < 0) | (pairs >= length)).any(dim=1)
        bad = pairs[mask_row][0]
        raise ValueError(f"Pair ({int(bad[0].item())}, {int(bad[1].item())}) is out of bounds for length {length}.")

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    contact_map_t[i_idx, j_idx] = True
    contact_map_t[j_idx, i_idx] = True
    return contact_map_t


def _numpy_pairs_to_contact_map(pairs: np.ndarray, length: int | None, unsafe: bool) -> np.ndarray:
    if pairs.size == 0:
        max_index = -1
    else:
        pairs = _numpy_normalize_pairs_low_high(pairs)
        max_index = int(np.max(pairs))

    if length is None:
        length = max_index + 1 if max_index >= 0 else 0

    contact_map = np.zeros((length, length), dtype=bool)
    if pairs.size == 0:
        return contact_map

    if np.any(pairs[:, 0] == pairs[:, 1]):
        if not unsafe:
            raise ValueError("Self-pairing (i == j) is not allowed in pairs.")
        warn("Self-pairing (i == j) is not allowed in pairs.\nIgnoring such pairs.")
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.size == 0:
        return contact_map

    if np.any((pairs < 0) | (pairs >= length)):
        mask_row = ((pairs < 0) | (pairs >= length)).any(axis=1)
        bad = pairs[np.flatnonzero(mask_row)[0]]
        raise ValueError(f"Pair ({bad[0]}, {bad[1]}) is out of bounds for length {length}.")

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    contact_map[i_idx, j_idx] = True
    contact_map[j_idx, i_idx] = True
    return contact_map


@overload
def contact_map_to_pairs(contact_map: Tensor, unsafe: bool = False, *, threshold: float = 0.5) -> Tensor: ...


@overload
def contact_map_to_pairs(  # type: ignore[overload-cannot-match]
    contact_map: np.ndarray,
    unsafe: bool = False,
    *,
    threshold: float = 0.5,
) -> np.ndarray: ...


@overload
def contact_map_to_pairs(  # type: ignore[overload-cannot-match]
    contact_map: Sequence,
    unsafe: bool = False,
    *,
    threshold: float = 0.5,
) -> PairsList: ...


def contact_map_to_pairs(
    contact_map: Tensor | np.ndarray | Sequence, unsafe: bool = False, *, threshold: float = 0.5
) -> Tensor | np.ndarray | PairsList:
    """
    Convert a contact map to a list of base pairs.

    If ``contact_map`` is a torch tensor, returns a ``(K, 2)`` torch.LongTensor.
    Otherwise, returns a numpy ``(K, 2)`` int array (list inputs return a list of tuples).

    For integer/bool contact maps, any non-zero entry is treated as a contact and the map is
    expected to represent a binary (symmetric) adjacency matrix.

    For floating-point contact maps, values are interpreted as pairing probabilities in ``[0, 1]``
    (or logits/scores in ``unsafe`` mode), and pairs are decoded using a greedy NMS-style
    one-to-one matching that prioritizes higher scores above ``threshold``.

    Examples:
        Torch input
        >>> import torch
        >>> contact_map_tensor = torch.tensor([[0, 0, 0, 1],
        ...                                   [0, 0, 1, 0],
        ...                                   [0, 1, 0, 0],
        ...                                   [1, 0, 0, 0]])
        >>> contact_map_to_pairs(contact_map_tensor).tolist()
        [[0, 3], [1, 2]]

        NumPy input
        >>> import numpy as np
        >>> contact_map_array = np.array([[0, 0, 0, 1],
        ...                               [0, 0, 1, 0],
        ...                               [0, 1, 0, 0],
        ...                               [1, 0, 0, 0]])
        >>> contact_map_to_pairs(contact_map_array).tolist()
        [[0, 3], [1, 2]]
        >>> contact_map_to_pairs(np.array([[0.0, 0.8], [0.8, 0.0]]), threshold=0.5).tolist()
        [[0, 1]]

        List input
        >>> contact_map_to_pairs([[0, 1], [1, 0]])
        [(0, 1)]
    """
    if isinstance(contact_map, Tensor):
        if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
            raise ValueError("Contact map must be a square 2D matrix.")
        if contact_map.is_floating_point():
            return _torch_contact_map_to_pairs_float(contact_map, unsafe=unsafe, threshold=threshold)
        return _torch_contact_map_to_pairs_binary(contact_map, unsafe=unsafe)
    if isinstance(contact_map, np.ndarray):
        if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
            raise ValueError("Contact map must be a square 2D matrix.")
        if np.issubdtype(contact_map.dtype, np.floating):
            return _numpy_contact_map_to_pairs_float(contact_map, unsafe=unsafe, threshold=threshold)
        return _numpy_contact_map_to_pairs_binary(contact_map, unsafe=unsafe)
    if isinstance(contact_map, Sequence):
        contact_map = np.asarray(contact_map)
        if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
            raise ValueError("Contact map must be a square 2D matrix.")
        if np.issubdtype(contact_map.dtype, np.floating):
            pairs = _numpy_contact_map_to_pairs_float(contact_map, unsafe=unsafe, threshold=threshold)
        else:
            pairs = _numpy_contact_map_to_pairs_binary(contact_map, unsafe=unsafe)
        return [tuple(pair) for pair in pairs.tolist()]
    raise TypeError("contact_map must be a torch.Tensor, numpy.ndarray, or sequence")


def _torch_contact_map_to_pairs_binary(contact_map: Tensor, unsafe: bool) -> Tensor:
    contact_map_bool = contact_map != 0
    n = contact_map_bool.shape[0]
    if not torch.equal(contact_map_bool, contact_map_bool.T):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        lower_all_zero = not bool(torch.any(torch.tril(contact_map_bool, diagonal=-1)).item())
        upper_all_zero = not bool(torch.any(torch.triu(contact_map_bool, diagonal=1)).item())
        if lower_all_zero != upper_all_zero:
            warn("Contact map is not symmetric.\nUsing the populated triangle to symmetrize.")
            tri = (
                torch.triu(contact_map_bool, diagonal=1)
                if lower_all_zero
                else torch.tril(contact_map_bool, diagonal=-1)
            )
            contact_map_bool = tri | tri.T
        else:
            warn("Contact map is not symmetric.\nSymmetrizing with `contact_map | contact_map.T`.")
            contact_map_bool = contact_map_bool | contact_map_bool.T

    if torch.any(torch.diag(contact_map_bool)):
        if not unsafe:
            raise ValueError(
                "Contact map diagonal must be zero (bases cannot pair with themselves).\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn("Contact map diagonal is not zero (bases cannot pair with themselves).\nSetting diagonal to zero.")
        contact_map_bool = contact_map_bool.clone()
        contact_map_bool.fill_diagonal_(False)

    row_sums = torch.count_nonzero(contact_map_bool, dim=1)
    multiple_pairings = torch.nonzero(row_sums > 1, as_tuple=False).squeeze(-1)
    if multiple_pairings.numel() > 0:
        if not unsafe:
            raise ValueError(
                f"Positions {multiple_pairings.tolist()} are paired to multiple other positions.\n"
                "Each base can only pair with at most one other base.\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn(
            f"Positions {multiple_pairings.tolist()} are paired to multiple other positions.\n"
            "Using only the first pairing occurrence for each position."
        )

    ii, jj = torch.where(torch.triu(contact_map_bool, diagonal=1))
    if multiple_pairings.numel() == 0:
        return torch.stack([ii.to(torch.long), jj.to(torch.long)], dim=1)

    if ii.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=contact_map_bool.device)
    order = torch.argsort(ii * n + jj)
    ii = ii[order].to(torch.long)
    jj = jj[order].to(torch.long)
    return _torch_greedy_match(ii, jj, n)


def _numpy_contact_map_to_pairs_binary(contact_map: np.ndarray, unsafe: bool) -> np.ndarray:
    contact_map_bool = contact_map != 0
    n = contact_map_bool.shape[0]
    if not np.array_equal(contact_map_bool, contact_map_bool.T):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        lower_all_zero = not np.any(np.tril(contact_map_bool, k=-1))
        upper_all_zero = not np.any(np.triu(contact_map_bool, k=1))
        if lower_all_zero != upper_all_zero:
            warn("Contact map is not symmetric.\nUsing the populated triangle to symmetrize.")
            tri = np.triu(contact_map_bool, k=1) if lower_all_zero else np.tril(contact_map_bool, k=-1)
            contact_map_bool = tri | tri.T
        else:
            warn("Contact map is not symmetric.\nSymmetrizing with `contact_map | contact_map.T`.")
            contact_map_bool = contact_map_bool | contact_map_bool.T

    if np.any(np.diag(contact_map_bool)):
        if not unsafe:
            raise ValueError(
                "Contact map diagonal must be zero (bases cannot pair with themselves).\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn("Contact map diagonal is not zero (bases cannot pair with themselves).\nSetting diagonal to zero.")
        np.fill_diagonal(contact_map_bool, False)

    row_sums = np.count_nonzero(contact_map_bool, axis=1)
    multiple_pairings = np.where(row_sums > 1)[0]
    if len(multiple_pairings) > 0:
        if not unsafe:
            raise ValueError(
                f"Positions {multiple_pairings.tolist()} are paired to multiple other positions.\n"
                "Each base can only pair with at most one other base.\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn(
            f"Positions {multiple_pairings.tolist()} are paired to multiple other positions.\n"
            "Using only the first pairing occurrence for each position."
        )

    if not len(multiple_pairings):
        ii, jj = np.where(np.triu(contact_map_bool, k=1))
        return np.column_stack((ii, jj))

    ii, jj = np.where(np.triu(contact_map_bool, k=1))
    if ii.size == 0:
        return np.empty((0, 2), dtype=int)
    ord_idx = np.lexsort((jj, ii))
    ii = ii[ord_idx]
    jj = jj[ord_idx]
    return _numpy_greedy_match(ii, jj, n)


def _torch_contact_map_to_pairs_float(contact_map: Tensor, unsafe: bool, threshold: float) -> Tensor:
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be between 0 and 1, but got {threshold}.")

    contact_map_values = contact_map
    n = contact_map_values.shape[0]
    device = contact_map_values.device

    if bool(torch.isnan(contact_map_values).any().item()):
        raise ValueError("Contact map contains NaN values; unable to decode base pairs.")

    lower_all_zero = not bool(torch.any(torch.tril(contact_map_values, diagonal=-1)).item())
    upper_all_zero = not bool(torch.any(torch.triu(contact_map_values, diagonal=1)).item())
    if lower_all_zero != upper_all_zero:
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        warn("Contact map is not symmetric.\nUsing the populated triangle to symmetrize.")
        contact_map_values = contact_map_values + contact_map_values.T

    upper = torch.triu(contact_map_values, diagonal=1)
    lower = torch.tril(contact_map_values, diagonal=-1)
    out_of_range = bool(((upper < 0) | (upper > 1)).any().item() or ((lower < 0) | (lower > 1)).any().item())
    if out_of_range:
        if not unsafe:
            min_val = float(torch.minimum(upper.min(), lower.min()).item())
            max_val = float(torch.maximum(upper.max(), lower.max()).item())
            raise ValueError(
                "Floating-point contact maps are expected to contain probabilities in [0, 1].\n"
                f"Got min={min_val}, max={max_val} (excluding the diagonal).\n"
                "Values outside [0, 1] look like logits or unbounded scores.\n"
                "Pass `unsafe=True` to apply a sigmoid automatically."
            )
        warn("Contact map values are outside [0, 1]. Applying sigmoid.")
        contact_map_values = contact_map_values.sigmoid()

    if not torch.allclose(contact_map_values, contact_map_values.T, rtol=1e-5, atol=1e-6):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` to symmetrize by averaging.")
        warn("Contact map is not symmetric.\nSymmetrizing by averaging with its transpose.")
        contact_map_values = (contact_map_values + contact_map_values.T) / 2

    if n == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    score_matrix = contact_map_values.clone()
    contact_map_bool = contact_map_values >= threshold

    if torch.any(torch.diag(contact_map_bool)):
        if not unsafe:
            raise ValueError("Diagonal must be zero (self-pairing not allowed). Pass unsafe=True to fix.")
        warn("Diagonal is non-zero, setting to zero (self-pairing not allowed).")
        contact_map_bool = contact_map_bool.clone()
        contact_map_bool.fill_diagonal_(False)
        score_matrix.fill_diagonal_(0)

    row_sums = torch.count_nonzero(contact_map_bool, dim=1)
    multiple_pairings = torch.nonzero(row_sums > 1, as_tuple=False).squeeze(-1)
    if multiple_pairings.numel() > 0:
        if not unsafe:
            raise ValueError(
                f"Multiple pairings detected at positions {multiple_pairings.tolist()}. "
                "Each base can pair with at most one other base. Pass unsafe=True to resolve using scores."
            )
        warn(f"Multiple pairings at positions {multiple_pairings.tolist()}, selecting highest scores.")

    ii, jj = torch.where(torch.triu(contact_map_bool, diagonal=1))
    if multiple_pairings.numel() == 0:
        return torch.stack([ii.to(torch.long), jj.to(torch.long)], dim=1)

    if ii.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=contact_map_bool.device)

    pair_scores = score_matrix[ii, jj]
    order = torch.argsort(pair_scores, descending=True)
    ii = ii[order].to(torch.long)
    jj = jj[order].to(torch.long)
    return _torch_greedy_match(ii, jj, n)


def _numpy_contact_map_to_pairs_float(contact_map: np.ndarray, unsafe: bool, threshold: float) -> np.ndarray:
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be between 0 and 1, but got {threshold}.")

    contact_map_values = contact_map.astype(float, copy=False)
    n = contact_map_values.shape[0]

    if np.isnan(contact_map_values).any():
        raise ValueError("Contact map contains NaN values; unable to decode base pairs.")

    lower_all_zero = not np.any(np.tril(contact_map_values, k=-1))
    upper_all_zero = not np.any(np.triu(contact_map_values, k=1))
    if lower_all_zero != upper_all_zero:
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        warn("Contact map is not symmetric.\nUsing the populated triangle to symmetrize.")
        contact_map_values = contact_map_values + contact_map_values.T

    upper = np.triu(contact_map_values, k=1)
    lower = np.tril(contact_map_values, k=-1)
    out_of_range = np.any((upper < 0) | (upper > 1)) or np.any((lower < 0) | (lower > 1))
    if out_of_range:
        if not unsafe:
            min_val = float(min(np.min(upper), np.min(lower)))
            max_val = float(max(np.max(upper), np.max(lower)))
            raise ValueError(
                "Floating-point contact maps are expected to contain probabilities in [0, 1].\n"
                f"Got min={min_val}, max={max_val} (excluding the diagonal).\n"
                "Values outside [0, 1] look like logits or unbounded scores.\n"
                "Pass `unsafe=True` to apply a sigmoid automatically."
            )
        warn("Contact map values are outside [0, 1]. Applying sigmoid.")
        with np.errstate(over="ignore"):
            contact_map_values = 1 / (1 + np.exp(-contact_map_values))

    if not np.allclose(contact_map_values, contact_map_values.T, rtol=1e-5, atol=1e-6):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` to symmetrize by averaging.")
        warn("Contact map is not symmetric.\nSymmetrizing by averaging with its transpose.")
        contact_map_values = (contact_map_values + contact_map_values.T) / 2

    if n == 0:
        return np.empty((0, 2), dtype=int)

    score_matrix = contact_map_values.copy()
    contact_map_bool = contact_map_values >= threshold

    if np.any(np.diag(contact_map_bool)):
        if not unsafe:
            raise ValueError("Diagonal must be zero (self-pairing not allowed). Pass unsafe=True to fix.")
        warn("Diagonal is non-zero, setting to zero (self-pairing not allowed).")
        np.fill_diagonal(contact_map_bool, False)
        np.fill_diagonal(score_matrix, 0)

    row_sums = np.count_nonzero(contact_map_bool, axis=1)
    multiple_pairings = np.where(row_sums > 1)[0]
    if len(multiple_pairings) > 0:
        if not unsafe:
            raise ValueError(
                f"Multiple pairings detected at positions {multiple_pairings.tolist()}. "
                "Each base can pair with at most one other base. Pass unsafe=True to resolve using scores."
            )
        warn(f"Multiple pairings at positions {multiple_pairings.tolist()}, selecting highest scores.")

    if not len(multiple_pairings):
        ii, jj = np.where(np.triu(contact_map_bool, k=1))
        return np.column_stack((ii, jj))

    ii, jj = np.where(np.triu(contact_map_bool, k=1))
    if ii.size == 0:
        return np.empty((0, 2), dtype=int)

    pair_scores = score_matrix[ii, jj]
    ord_idx = np.argsort(-pair_scores)
    ii = ii[ord_idx]
    jj = jj[ord_idx]
    return _numpy_greedy_match(ii, jj, n)


def _torch_greedy_match(ii: Tensor, jj: Tensor, length: int) -> Tensor:
    if ii.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=ii.device)
    used = torch.zeros(length, dtype=torch.bool, device=ii.device)
    keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
    for k in range(ii.shape[0]):
        a = ii[k]
        b = jj[k]
        can_take = (~used[a]) & (~used[b])
        keep[k] = can_take
        used[a] = used[a] | can_take
        used[b] = used[b] | can_take
    out = torch.stack([ii[keep], jj[keep]], dim=1)
    if out.numel() == 0:
        return out
    key = out[:, 0] * length + out[:, 1]
    order = torch.argsort(key)
    return out[order]


def _numpy_greedy_match(ii: np.ndarray, jj: np.ndarray, length: int) -> np.ndarray:
    if ii.size == 0:
        return np.empty((0, 2), dtype=int)
    used = np.zeros(length, dtype=bool)
    keep = np.zeros(ii.shape[0], dtype=bool)
    for idx in range(ii.shape[0]):
        i = ii[idx]
        j = jj[idx]
        if not used[i] and not used[j]:
            keep[idx] = True
            used[i] = True
            used[j] = True
    out = np.column_stack((ii[keep], jj[keep]))
    if out.size == 0:
        return out
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def pairs_to_dot_bracket(
    pairs: Tensor | np.ndarray | Pairs,
    length: int,
    unsafe: bool = False,
) -> str:
    """
    Convert base pairs to a dot-bracket string (backend-aware input, string output).

    Torch inputs are accepted and internally converted to NumPy for string building.
    In safe mode, tiers are assigned using an exact minimal-tier coloring.
    In unsafe mode, a greedy tiering is used for speed and may use more bracket types.

    Examples:
        Torch input
        >>> import torch
        >>> pairs_to_dot_bracket(torch.tensor([[0, 2], [1, 3]]), length=4)
        '([)]'

        NumPy input
        >>> import numpy as np
        >>> pairs_to_dot_bracket(np.array([(0, 3), (1, 2)]), length=4)
        '(())'

        List input
        >>> pairs_to_dot_bracket([(0, 3), (1, 2)], length=4)
        '(())'
    """
    # Always operate in NumPy for string construction
    if isinstance(pairs, Tensor):
        pairs = pairs.detach().cpu().numpy()
    elif isinstance(pairs, np.ndarray):
        pass
    elif isinstance(pairs, Sequence):
        pairs = np.asarray(list(pairs), dtype=int)
    else:
        raise TypeError("pairs must be a torch.Tensor, numpy.ndarray, or sequence of (i, j) pairs")
    if pairs.size == 0:
        return _numpy_pairs_to_dot_bracket(pairs, length, unsafe)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be an array-like with shape (n, 2)")
    return _numpy_pairs_to_dot_bracket(pairs, length, unsafe)


def _numpy_pairs_to_dot_bracket(pairs: np.ndarray, length: int, unsafe: bool) -> str:
    if pairs.size == 0:
        return "." * length
    pairs = _numpy_normalize_pairs_low_high(pairs)
    i = pairs[:, 0]
    j = pairs[:, 1]
    self_mask = i == j
    if np.any(self_mask) and not unsafe:
        raise ValueError("Self-pairing (i == j) is invalid.")
    if np.any(self_mask) and unsafe:
        warn("Ignoring self-pairing (i == j) in pairs_to_dot_bracket.")
    in_bounds = (0 <= i) & (i < length) & (0 <= j) & (j < length)
    keep = in_bounds & (~self_mask)
    if np.any(~in_bounds) and unsafe:
        bad = np.column_stack((i[~in_bounds], j[~in_bounds]))
        if bad.size:
            warn(f"Ignoring out-of-bounds pairs {bad.tolist()} for length {length}.")
    if np.any(~in_bounds) and not unsafe:
        idx = int(np.flatnonzero(~in_bounds)[0])
        raise ValueError(f"Pair ({int(i[idx])}, {int(j[idx])}) is out of bounds for length {length}.")
    pairs = np.column_stack((i[keep], j[keep]))

    seen = np.zeros(length, dtype=bool)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    keep_idx: List[int] = []
    dup_positions: List[int] = []
    for idx in range(pairs.shape[0]):
        a, b = int(pairs[idx, 0]), int(pairs[idx, 1])
        if seen[a] or seen[b]:
            if unsafe:
                dup_positions.extend([a, b])
                continue
            raise ValueError(f"Positions {a} or {b} are paired multiple times; not allowed.")
        seen[a] = True
        seen[b] = True
        keep_idx.append(idx)
    if dup_positions and unsafe:
        warn(
            f"Some positions appear in multiple pairs: {sorted(set(dup_positions))}.\n"
            "Keeping only the first occurrence."
        )
    pairs = pairs[keep_idx] if keep_idx else np.empty((0, 2), dtype=int)

    dot_bracket = ["." for _ in range(length)]
    bracket_types = list(_DOT_BRACKET_PAIR_TABLE.items())
    tiers = pseudoknot_tiers(pairs, unsafe=unsafe)

    if len(tiers) > len(bracket_types):
        if not unsafe:
            raise ValueError("Could not represent all base pairs with available bracket types.")
        warn(
            "Too many pseudoknot tiers; could not represent all base pairs with available bracket types.\n"
            f"Omitting {sum(len(t) for t in tiers[len(bracket_types):])} pairs."
        )
        tiers = tiers[: len(bracket_types)]

    for tier, (open_bracket, close_bracket) in zip(tiers, bracket_types):
        for a, b in tier:
            dot_bracket[a] = open_bracket
            dot_bracket[b] = close_bracket

    return "".join(dot_bracket)


def dot_bracket_to_contact_map(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a numpy contact map.

    Examples:
        >>> dot_bracket_to_contact_map('(())').astype(int).tolist()
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    """
    return pairs_to_contact_map(dot_bracket_to_pairs(dot_bracket), length=len(dot_bracket))


def contact_map_to_dot_bracket(
    contact_map: Tensor | np.ndarray, unsafe: bool = False, *, threshold: float = 0.5
) -> str:
    """
    Convert a contact map (NumPy or Torch) to a dot-bracket notation string.

    Examples:
        Torch input
        >>> import torch
        >>> contact_map_tensor = torch.tensor([[0, 0, 0, 1],
        ...                                    [0, 0, 1, 0],
        ...                                    [0, 1, 0, 0],
        ...                                    [1, 0, 0, 0]])
        >>> contact_map_to_dot_bracket(contact_map_tensor)
        '(())'

        NumPy input
        >>> import numpy as np
        >>> contact_map = np.array([[0, 0, 0, 1],
        ...                          [0, 0, 1, 0],
        ...                          [0, 1, 0, 0],
        ...                          [1, 0, 0, 0]])
        >>> contact_map_to_dot_bracket(contact_map)
        '(())'

        List input
        >>> contact_map_to_dot_bracket([[0, 1], [1, 0]])
        '()'
    """
    return pairs_to_dot_bracket(
        contact_map_to_pairs(contact_map, unsafe=unsafe, threshold=threshold), length=len(contact_map), unsafe=unsafe
    )
