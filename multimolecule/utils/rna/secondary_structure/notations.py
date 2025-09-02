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
from collections.abc import Mapping
from typing import Dict, List, Sequence, Tuple
from warnings import warn

import numpy as np
import torch
from torch import Tensor

_DOT_BRACKET_PAIR_TABLE: Dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
_DOT_BRACKET_PAIR_TABLE.update(zip(string.ascii_uppercase, string.ascii_lowercase))
_REVERSE_DOT_BRACKET_PAIR_TABLE: Mapping[str, str] = {v: k for k, v in _DOT_BRACKET_PAIR_TABLE.items()}
_UNPAIRED_TOKENS = {"+", ".", ",", "_"}


def dot_bracket_to_pairs(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a list of base-pair indices.

    Args:
        dot_bracket: Dot-bracket notation. Supports pseudoknots via multiple
            bracket types, including (), [], {}, <>, and A-Z/a-z.

    Returns:
        A numpy array of shape (n, 2) with pairs ``(i, j)`` where ``0 <= i < j < len(dot_bracket)``.

    Raises:
        ValueError: On unmatched or invalid symbols.

    Examples:
        >>> dot_bracket_to_pairs("((.))")
        array([[0, 4],
               [1, 3]])
        >>> dot_bracket_to_pairs("...")
        array([], shape=(0, 2), dtype=int64)
    """
    stacks: defaultdict[str, List[int]] = defaultdict(list)
    pairs: List[Tuple[int, int]] = []
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


def pairs_to_contact_map(
    pairs: Tensor | np.ndarray | Sequence[Tuple[int, int]],
    length: int | None = None,
    unsafe: bool = False,
) -> Tensor | np.ndarray:
    """
    Convert base pairs to a symmetric contact map.

    If ``pairs`` is a torch tensor, returns a boolean torch.Tensor on the same device.
    Otherwise, returns a numpy boolean array.

    Examples:
        NumPy input
        >>> import numpy as np
        >>> cm = pairs_to_contact_map(np.array([(0, 3), (1, 2)]), length=4)
        >>> cm.astype(int)
        array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [1, 0, 0, 0]])

        Torch input
        >>> import torch
        >>> tcm = pairs_to_contact_map(torch.tensor([[0, 3], [1, 2]]), length=4)
        >>> tcm.to(torch.int).tolist()
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    """
    if isinstance(pairs, Tensor):
        return _torch_pairs_to_contact_map(pairs, length, unsafe)
    if not isinstance(pairs, np.ndarray):
        pairs = np.asarray(pairs, dtype=int)
    return _numpy_pairs_to_contact_map(pairs, length, unsafe)


def _torch_pairs_to_contact_map(pairs: Tensor, length: int | None, unsafe: bool) -> Tensor:
    if pairs.numel() == 0:
        max_index = -1
        device = pairs.device
    else:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a sequence/array of (i, j) index tuples")
        pairs = pairs.to(dtype=torch.long)
        low = torch.minimum(pairs[:, 0], pairs[:, 1])
        high = torch.maximum(pairs[:, 0], pairs[:, 1])
        pairs = torch.stack([low, high], dim=1)
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
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a sequence/array of (i, j) index tuples")
        pairs = pairs.astype(int, copy=False)
        low = np.minimum(pairs[:, 0], pairs[:, 1])
        high = np.maximum(pairs[:, 0], pairs[:, 1])
        pairs = np.stack([low, high], axis=1)
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


def contact_map_to_pairs(
    contact_map: Tensor | np.ndarray, unsafe: bool = False, *, threshold: float = 0.5
) -> Tensor | np.ndarray:
    """
    Convert a contact map to a list of base pairs.

    If ``contact_map`` is a torch tensor, returns a ``(K, 2)`` torch.LongTensor.
    Otherwise, returns a numpy ``(K, 2)`` int array.

    For integer/bool contact maps, any non-zero entry is treated as a contact and the map is
    expected to represent a binary (symmetric) adjacency matrix.

    For floating-point contact maps, values are interpreted as pairing probabilities in ``[0, 1]``
    (or logits/scores in ``unsafe`` mode), and pairs are decoded using a mutual row/column argmax
    selection above ``threshold`` (an NMS-like one-to-one matching).

    Examples:
        NumPy input
        >>> import numpy as np
        >>> cm = np.array([[0, 0, 0, 1],
        ...                [0, 0, 1, 0],
        ...                [0, 1, 0, 0],
        ...                [1, 0, 0, 0]])
        >>> contact_map_to_pairs(cm)
        array([[0, 3],
               [1, 2]])

        Torch input
        >>> import torch
        >>> tcm = torch.tensor([[0, 0, 0, 1],
        ...                     [0, 0, 1, 0],
        ...                     [0, 1, 0, 0],
        ...                     [1, 0, 0, 0]])
        >>> contact_map_to_pairs(tcm).tolist()
        [[0, 3], [1, 2]]
    """
    if isinstance(contact_map, Tensor):
        if contact_map.is_floating_point():
            return _torch_contact_map_to_pairs_float(contact_map, unsafe=unsafe, threshold=threshold)
        return _torch_contact_map_to_pairs_binary(contact_map, unsafe=unsafe)
    if not isinstance(contact_map, np.ndarray):
        contact_map = np.asarray(contact_map)
    if np.issubdtype(contact_map.dtype, np.floating):
        return _numpy_contact_map_to_pairs_float(contact_map, unsafe=unsafe, threshold=threshold)
    return _numpy_contact_map_to_pairs_binary(contact_map, unsafe=unsafe)


def _torch_contact_map_to_pairs_binary(contact_map: Tensor, unsafe: bool) -> Tensor:
    cm = contact_map != 0
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    n = cm.shape[0]
    if not torch.equal(cm, cm.T):
        lower_all_zero = not bool(torch.any(torch.tril(cm, diagonal=-1)).item())
        upper_all_zero = not bool(torch.any(torch.triu(cm, diagonal=1)).item())
        if lower_all_zero != upper_all_zero:
            tri = torch.triu(cm, diagonal=1) if lower_all_zero else torch.tril(cm, diagonal=-1)
            cm = tri | tri.T
        elif not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        else:
            warn("Contact map is not symmetric.\nSymmetrizing with `cm | cm.T`.")
            cm = cm | cm.T

    if torch.any(torch.diag(cm)):
        if not unsafe:
            raise ValueError(
                "Contact map diagonal must be zero (bases cannot pair with themselves).\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn("Contact map diagonal is not zero (bases cannot pair with themselves).\nSetting diagonal to zero.")
        cm = cm.clone()
        cm.fill_diagonal_(False)

    row_sums = torch.count_nonzero(cm, dim=1)
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

    ii, jj = torch.where(torch.triu(cm, diagonal=1))
    if multiple_pairings.numel() == 0:
        return torch.stack([ii.to(torch.long), jj.to(torch.long)], dim=1)

    if ii.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=cm.device)
    order = torch.argsort(ii * n + jj)
    ii = ii[order]
    jj = jj[order]
    used = torch.zeros(n, dtype=torch.bool, device=cm.device)
    keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=cm.device)
    for k in range(ii.shape[0]):
        a = int(ii[k].item())
        b = int(jj[k].item())
        if not used[a] and not used[b]:
            keep[k] = True
            used[a] = True
            used[b] = True
    return torch.stack([ii[keep].to(torch.long), jj[keep].to(torch.long)], dim=1)


def _numpy_contact_map_to_pairs_binary(contact_map: np.ndarray, unsafe: bool) -> np.ndarray:
    cm = contact_map != 0
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    n = cm.shape[0]
    if not np.array_equal(cm, cm.T):
        lower_all_zero = not np.any(np.tril(cm, k=-1))
        upper_all_zero = not np.any(np.triu(cm, k=1))
        if lower_all_zero != upper_all_zero:
            tri = np.triu(cm, k=1) if lower_all_zero else np.tril(cm, k=-1)
            cm = tri | tri.T
        elif not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        else:
            warn("Contact map is not symmetric.\nSymmetrizing with `cm | cm.T`.")
            cm = cm | cm.T

    if np.any(np.diag(cm)):
        if not unsafe:
            raise ValueError(
                "Contact map diagonal must be zero (bases cannot pair with themselves).\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn("Contact map diagonal is not zero (bases cannot pair with themselves).\nSetting diagonal to zero.")

        np.fill_diagonal(cm, False)

    row_sums = np.count_nonzero(cm, axis=1)
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
        ii, jj = np.where(np.triu(cm, k=1))
        return np.column_stack((ii, jj))

    ii, jj = np.where(np.triu(cm, k=1))
    if ii.size == 0:
        return np.empty((0, 2), dtype=int)
    ord_idx = np.lexsort((jj, ii))
    ii = ii[ord_idx]
    jj = jj[ord_idx]
    used = np.zeros(n, dtype=bool)
    keep = np.zeros(ii.shape[0], dtype=bool)
    for idx in range(ii.shape[0]):
        i = ii[idx]
        j = jj[idx]
        if not used[i] and not used[j]:
            keep[idx] = True
            used[i] = True
            used[j] = True
    return np.column_stack((ii[keep], jj[keep]))


def _torch_contact_map_to_pairs_float(contact_map: Tensor, unsafe: bool, threshold: float) -> Tensor:
    if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be between 0 and 1, but got {threshold}.")

    cm = contact_map
    n = cm.shape[0]
    device = cm.device

    if bool(torch.isnan(cm).any().item()):
        raise ValueError("Contact map contains NaN values; unable to decode base pairs.")

    lower_all_zero = not bool(torch.any(torch.tril(cm, diagonal=-1)).item())
    upper_all_zero = not bool(torch.any(torch.triu(cm, diagonal=1)).item())
    if lower_all_zero != upper_all_zero:
        cm = cm + cm.T

    upper = torch.triu(cm, diagonal=1)
    lower = torch.tril(cm, diagonal=-1)
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
        cm = cm.sigmoid()

    if not torch.allclose(cm, cm.T, rtol=1e-5, atol=1e-6):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` to symmetrize by averaging.")
        warn("Contact map is not symmetric.\nSymmetrizing by averaging with its transpose.")
        cm = (cm + cm.T) / 2

    if n == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    masked = cm.clone()
    masked.fill_diagonal_(float("-inf"))
    masked[masked <= threshold] = float("-inf")

    row_max_indices = torch.argmax(masked, dim=1)
    col_max_indices = torch.argmax(masked, dim=0)
    row_max_values = masked[torch.arange(n, device=device), row_max_indices]
    col_max_values = masked[col_max_indices, torch.arange(n, device=device)]

    has_valid_row_pair = row_max_values > threshold
    has_valid_col_pair = col_max_values > threshold
    mutual_selection = col_max_indices[row_max_indices] == torch.arange(n, device=device)
    mutual_pairs = has_valid_row_pair & has_valid_col_pair[row_max_indices] & mutual_selection

    ii = torch.nonzero(mutual_pairs, as_tuple=False).squeeze(-1)
    if ii.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    jj = row_max_indices[ii]
    keep = ii < jj
    if not bool(keep.any().item()):
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.stack([ii[keep].to(torch.long), jj[keep].to(torch.long)], dim=1)


def _numpy_contact_map_to_pairs_float(contact_map: np.ndarray, unsafe: bool, threshold: float) -> np.ndarray:
    if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be between 0 and 1, but got {threshold}.")

    cm = contact_map.astype(float, copy=False)
    n = cm.shape[0]

    if np.isnan(cm).any():
        raise ValueError("Contact map contains NaN values; unable to decode base pairs.")

    lower_all_zero = not np.any(np.tril(cm, k=-1))
    upper_all_zero = not np.any(np.triu(cm, k=1))
    if lower_all_zero != upper_all_zero:
        cm = cm + cm.T

    upper = np.triu(cm, k=1)
    lower = np.tril(cm, k=-1)
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
            cm = 1 / (1 + np.exp(-cm))

    if not np.allclose(cm, cm.T, rtol=1e-5, atol=1e-6):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` to symmetrize by averaging.")
        warn("Contact map is not symmetric.\nSymmetrizing by averaging with its transpose.")
        cm = (cm + cm.T) / 2

    if n == 0:
        return np.empty((0, 2), dtype=int)

    masked = cm.copy()
    np.fill_diagonal(masked, -np.inf)
    masked[masked <= threshold] = -np.inf

    row_max_indices = np.argmax(masked, axis=1)
    col_max_indices = np.argmax(masked, axis=0)
    row_max_values = masked[np.arange(n), row_max_indices]
    col_max_values = masked[col_max_indices, np.arange(n)]

    has_valid_row_pair = row_max_values > threshold
    has_valid_col_pair = col_max_values > threshold
    mutual_selection = col_max_indices[row_max_indices] == np.arange(n)
    mutual_pairs = has_valid_row_pair & has_valid_col_pair[row_max_indices] & mutual_selection

    ii = np.where(mutual_pairs)[0]
    if ii.size == 0:
        return np.empty((0, 2), dtype=int)
    jj = row_max_indices[ii]
    keep = ii < jj
    if not np.any(keep):
        return np.empty((0, 2), dtype=int)
    return np.column_stack((ii[keep], jj[keep])).astype(int, copy=False)


def pairs_to_dot_bracket(
    pairs: Tensor | np.ndarray | Sequence[Tuple[int, int]],
    length: int,
    unsafe: bool = False,
) -> str:
    """
    Convert base pairs to a dot-bracket string (backend-aware input, string output).

    Torch inputs are accepted and internally converted to NumPy for string building.
    In safe mode, tiers are assigned using an exact minimal-tier coloring. In unsafe
    mode, a greedy tiering is used for speed and may use more bracket types.

    Examples:
        NumPy input
        >>> import numpy as np
        >>> pairs_to_dot_bracket(np.array([(0, 3), (1, 2)]), length=4)
        '(())'

        Torch input
        >>> import torch
        >>> pairs_to_dot_bracket(torch.tensor([[0, 2], [1, 3]]), length=4)
        '([)]'
    """
    # Always operate in NumPy for string construction
    if isinstance(pairs, Tensor):
        pairs_np = pairs.detach().cpu().numpy()
    elif isinstance(pairs, np.ndarray):
        pairs_np = pairs
    else:
        pairs_np = np.asarray(list(pairs), dtype=int)
    return _numpy_pairs_to_dot_bracket(pairs_np, length, unsafe)


def _numpy_pairs_to_dot_bracket(pairs: np.ndarray, length: int, unsafe: bool) -> str:
    if pairs.size == 0:
        return "." * length
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise TypeError("pairs must be a numpy.ndarray with shape (n, 2)")
    pairs = pairs.astype(int, copy=False)
    i = pairs[:, 0]
    j = pairs[:, 1]
    self_mask = i == j
    if np.any(self_mask) and not unsafe:
        raise ValueError("Self-pairing (i == j) is invalid.")
    if np.any(self_mask) and unsafe:
        warn("Ignoring self-pairing (i == j) in pairs_to_dot_bracket.")
    i0 = np.minimum(i, j)
    j0 = np.maximum(i, j)
    in_bounds = (0 <= i0) & (i0 < length) & (0 <= j0) & (j0 < length)
    keep = in_bounds & (~self_mask)
    if np.any(~in_bounds) and unsafe:
        bad = np.column_stack((i0[~in_bounds], j0[~in_bounds]))
        if bad.size:
            warn(f"Ignoring out-of-bounds pairs {bad.tolist()} for length {length}.")
    if np.any(~in_bounds) and not unsafe:
        idx = int(np.flatnonzero(~in_bounds)[0])
        raise ValueError(f"Pair ({int(i0[idx])}, {int(j0[idx])}) is out of bounds for length {length}.")
    normalized_arr = np.column_stack((i0[keep], j0[keep]))

    seen = np.zeros(length, dtype=bool)
    order = np.lexsort((normalized_arr[:, 1], normalized_arr[:, 0]))
    norm_sorted = normalized_arr[order]
    keep_idx: List[int] = []
    dup_positions: List[int] = []
    for idx in range(norm_sorted.shape[0]):
        a, b = int(norm_sorted[idx, 0]), int(norm_sorted[idx, 1])
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
    filtered_arr = norm_sorted[keep_idx] if keep_idx else np.empty((0, 2), dtype=int)

    dot_bracket = ["." for _ in range(length)]
    bracket_types = list(_DOT_BRACKET_PAIR_TABLE.items())
    # Use minimal-tier coloring in safe mode; greedy tiering in unsafe mode for speed.
    tiers = _greedy_pseudoknot_tiers(filtered_arr) if unsafe else _minimal_pseudoknot_tiers(filtered_arr)

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


def _minimal_pseudoknot_tiers(pairs: np.ndarray) -> List[List[Tuple[int, int]]]:
    if pairs.size == 0:
        return []
    adj = _crossing_adjacency(pairs)
    colors, num_colors = _dsatur_min_coloring(adj)
    tiers: List[List[Tuple[int, int]]] = [[] for _ in range(num_colors)]
    for (a, b), color in zip(pairs.tolist(), colors):
        tiers[color].append((a, b))
    return tiers


def _crossing_adjacency(pairs: np.ndarray) -> List[List[int]]:
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
            if (ai < bi < aj < bj) or (bi < ai < bj < aj):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _dsatur_min_coloring(adj: List[List[int]]) -> Tuple[List[int], int]:
    n = len(adj)
    if n == 0:
        return [], 0
    degrees = [len(neighbors) for neighbors in adj]
    best_colors, best = _dsatur_greedy_coloring(adj, degrees)
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
        node = _select_dsatur_vertex(uncolored, neighbor_color_counts, degrees)
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


def _dsatur_greedy_coloring(adj: List[List[int]], degrees: List[int]) -> Tuple[List[int], int]:
    n = len(adj)
    colors = [-1] * n
    neighbor_color_counts: List[Dict[int, int]] = [{} for _ in range(n)]
    uncolored = [True] * n
    remaining = n
    num_colors = 0

    while remaining:
        v = _select_dsatur_vertex(uncolored, neighbor_color_counts, degrees)
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


def _select_dsatur_vertex(
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


def _greedy_pseudoknot_tiers(pairs: np.ndarray) -> List[List[Tuple[int, int]]]:
    tiers: List[List[Tuple[int, int]]] = []
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


def dot_bracket_to_contact_map(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a numpy contact map.

    Examples:
        >>> dot_bracket_to_contact_map('(())').astype(int)
        array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [1, 0, 0, 0]])
    """
    return pairs_to_contact_map(dot_bracket_to_pairs(dot_bracket), length=len(dot_bracket))


def contact_map_to_dot_bracket(
    contact_map: Tensor | np.ndarray, unsafe: bool = False, *, threshold: float = 0.5
) -> str:
    """
    Convert a contact map (NumPy or Torch) to a dot-bracket notation string.

    Examples:
        NumPy input
        >>> import numpy as np
        >>> cm = np.array([[0, 0, 0, 1],
        ...                [0, 0, 1, 0],
        ...                [0, 1, 0, 0],
        ...                [1, 0, 0, 0]])
        >>> contact_map_to_dot_bracket(cm)
        '(())'

        Torch input
        >>> import torch
        >>> tcm = torch.tensor([[0, 0, 0, 1],
        ...                     [0, 0, 1, 0],
        ...                     [0, 1, 0, 0],
        ...                     [1, 0, 0, 0]])
        >>> contact_map_to_dot_bracket(tcm)
        '(())'
    """
    return pairs_to_dot_bracket(
        contact_map_to_pairs(contact_map, unsafe=unsafe, threshold=threshold), length=len(contact_map), unsafe=unsafe
    )
