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
from typing import List, Sequence, Tuple
from warnings import warn

import numpy as np
import torch
from torch import Tensor

_DOT_BRACKET_PAIR_TABLE: dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
_DOT_BRACKET_PAIR_TABLE.update(zip(string.ascii_uppercase, string.ascii_lowercase))
_REVERSE_DOT_BRACKET_PAIR_TABLE: dict[str, str] = {v: k for k, v in _DOT_BRACKET_PAIR_TABLE.items()}
_UNPAIRED_TOKENS = {"+", ".", ",", "_"}


def dot_bracket_to_pairs(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a list of base-pair indices.

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
    stacks: defaultdict[str, list[int]] = defaultdict(list)
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
    Convert base pairs to a symmetric contact map (backend-aware).

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
    return _np_pairs_to_contact_map(pairs, length, unsafe)


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
        bad = pairs[torch.nonzero(mask_row, as_tuple=False)[0]]
        raise ValueError(f"Pair ({int(bad[0].item())}, {int(bad[1].item())}) is out of bounds for length {length}.")

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    contact_map_t[i_idx, j_idx] = True
    contact_map_t[j_idx, i_idx] = True
    return contact_map_t


def _np_pairs_to_contact_map(pairs: np.ndarray, length: int | None, unsafe: bool) -> np.ndarray:
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
    contact_map: Tensor | np.ndarray | Sequence[Tuple[int, int]], unsafe: bool = False
) -> Tensor | np.ndarray:
    """
    Convert a contact map to a list of base pairs (backend-aware).

    If ``contact_map`` is a torch tensor, returns a ``(K, 2)`` torch.LongTensor.
    Otherwise, returns a numpy ``(K, 2)`` int array. Validates symmetry and zero diagonal
    unless ``unsafe=True``.

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
        return _torch_contact_map_to_pairs(contact_map, unsafe)
    if not isinstance(contact_map, np.ndarray):
        contact_map = np.asarray(contact_map)
    return _np_contact_map_to_pairs(contact_map, unsafe)


def _torch_contact_map_to_pairs(contact_map: Tensor, unsafe: bool) -> Tensor:
    cm = contact_map != 0
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    n = cm.shape[0]
    if not torch.equal(cm, cm.T):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        warn("Contact map is not symmetric.\nUsing only the upper triangular part.")
        triu = torch.triu(cm)
        cm = triu | triu.T

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


def _np_contact_map_to_pairs(contact_map: np.ndarray, unsafe: bool) -> np.ndarray:
    cm = contact_map != 0
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    n = cm.shape[0]
    if not np.array_equal(cm, cm.T):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        warn("Contact map is not symmetric.\nUsing only the upper triangular part.")
        triu = np.triu(cm)
        cm = triu | triu.T

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


def pairs_to_dot_bracket(
    pairs: Tensor | np.ndarray | Sequence[Tuple[int, int]],
    length: int,
    unsafe: bool = False,
) -> str:
    """
    Convert base pairs to a dot-bracket string (backend-aware input, string output).

    Torch inputs are accepted and internally converted to NumPy for string building.

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
    return _np_pairs_to_dot_bracket(pairs_np, length, unsafe)


def _np_pairs_to_dot_bracket(pairs: np.ndarray, length: int, unsafe: bool) -> str:
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

    tiers: list[list[Tuple[int, int]]] = []
    end_stacks: list[list[int]] = []
    for a, b in filtered_arr.tolist():
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
        >>> dot_bracket_to_contact_map('(())').astype(int)
        array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [1, 0, 0, 0]])
    """
    return pairs_to_contact_map(dot_bracket_to_pairs(dot_bracket), length=len(dot_bracket))


def contact_map_to_dot_bracket(
    contact_map: Tensor | np.ndarray | Sequence[Tuple[int, int]], unsafe: bool = False
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
        contact_map_to_pairs(contact_map, unsafe=unsafe), length=len(contact_map), unsafe=unsafe
    )


def pseudoknot_pairs(pairs: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """
    Return subset of base pairs that participate in any pseudoknot crossing (backend-aware).

    Examples:
        NumPy input
        >>> import numpy as np
        >>> pseudoknot_pairs(np.array([(0, 2), (1, 3)])).tolist()
        [[0, 2], [1, 3]]
        >>> pseudoknot_pairs(np.array([(0, 3), (1, 2)])).shape
        (0, 2)

        Torch input
        >>> import torch
        >>> pseudoknot_pairs(torch.tensor([[0, 2], [1, 3]])).tolist()
        [[0, 2], [1, 3]]
    """
    if isinstance(pairs, Tensor):
        return _torch_pseudoknot_pairs(pairs)
    if not isinstance(pairs, np.ndarray) or (pairs.size != 0 and (pairs.ndim != 2 or pairs.shape[1] != 2)):
        raise TypeError("pairs must be a numpy.ndarray with shape (n, 2)")
    return _np_pseudoknot_pairs(pairs)


def _torch_pseudoknot_pairs(pairs: Tensor) -> Tensor:
    if pairs.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise TypeError("pairs must have shape (n, 2)")
    i = torch.minimum(pairs[:, 0], pairs[:, 1]).to(torch.long)
    j = torch.maximum(pairs[:, 0], pairs[:, 1]).to(torch.long)
    norm = torch.stack([i, j], dim=1)
    norm = torch.unique(norm, dim=0)
    if norm.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    key = norm[:, 0] * (int(norm[:, 1].max().item()) + 1 if norm.numel() else 1) + norm[:, 1]
    norm = norm[torch.argsort(key)]
    n = norm.shape[0]
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    ii = norm[:, 0]
    jj = norm[:, 1]
    tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=norm.device), diagonal=1)
    crosses = tri & (ii.unsqueeze(0) < jj.unsqueeze(1)) & (jj.unsqueeze(1) < jj.unsqueeze(0))
    if not crosses.any():
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    pk_mask = crosses.any(dim=1) | crosses.any(dim=0)
    out = norm[pk_mask]
    return out.to(torch.long)


def _np_pseudoknot_pairs(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    i = np.minimum(pairs[:, 0], pairs[:, 1])
    j = np.maximum(pairs[:, 0], pairs[:, 1])
    norm = np.column_stack((i, j)).astype(int, copy=False)
    norm = np.unique(norm, axis=0)
    if norm.size == 0:
        return np.empty((0, 2), dtype=int)
    ord_idx = np.lexsort((norm[:, 1], norm[:, 0]))
    norm = norm[ord_idx]
    n = norm.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=int)
    ii = norm[:, 0]
    jj = norm[:, 1]
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    crosses = tri & (ii[None, :] < jj[:, None]) & (jj[:, None] < jj[None, :])
    if not crosses.any():
        return np.empty((0, 2), dtype=int)
    pk_mask = crosses.any(axis=1) | crosses.any(axis=0)
    out = norm[pk_mask]
    return out.astype(int, copy=False)


def pseudoknot_nucleotides(pairs: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """
    Return nucleotide indices involved in any pseudoknot pair (backend-aware).

    Examples:
        NumPy input
        >>> import numpy as np
        >>> pseudoknot_nucleotides(np.array([(0, 2), (1, 3)])).tolist()
        [0, 1, 2, 3]

        Torch input
        >>> import torch
        >>> pseudoknot_nucleotides(torch.tensor([[0, 2], [1, 3]])).tolist()
        [0, 1, 2, 3]
    """
    if isinstance(pairs, Tensor):
        return _torch_pseudoknot_nucleotides(pairs)
    if not isinstance(pairs, np.ndarray) or (pairs.size != 0 and (pairs.ndim != 2 or pairs.shape[1] != 2)):
        raise TypeError("pairs must be a numpy.ndarray with shape (n, 2)")
    return _np_pseudoknot_nucleotides(pairs)


def _torch_pseudoknot_nucleotides(pairs: Tensor) -> Tensor:
    pkp = _torch_pseudoknot_pairs(pairs)
    if pkp.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=pairs.device)
    return torch.unique(pkp.view(-1)).to(torch.long)


def _np_pseudoknot_nucleotides(pairs: np.ndarray) -> np.ndarray:
    pkp = _np_pseudoknot_pairs(pairs)
    if pkp.size == 0:
        return np.empty((0,), dtype=int)
    return np.unique(pkp.reshape(-1))


# Explicit re-exports (including private constants) for compatibility
__all__ = [
    "dot_bracket_to_pairs",
    "pairs_to_contact_map",
    "contact_map_to_pairs",
    "pairs_to_dot_bracket",
    "dot_bracket_to_contact_map",
    "contact_map_to_dot_bracket",
    "pseudoknot_pairs",
    "pseudoknot_nucleotides",
    "_DOT_BRACKET_PAIR_TABLE",
    "_REVERSE_DOT_BRACKET_PAIR_TABLE",
    "_UNPAIRED_TOKENS",
]
