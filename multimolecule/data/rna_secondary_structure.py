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
from torch import Tensor

_DOT_BRACKET_PAIR_TABLE: dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
_DOT_BRACKET_PAIR_TABLE.update(zip(string.ascii_uppercase, string.ascii_lowercase))
_REVERSE_DOT_BRACKET_PAIR_TABLE: dict[str, str] = {v: k for k, v in _DOT_BRACKET_PAIR_TABLE.items()}
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
    pairs: np.ndarray | Sequence[Tuple[int, int]], length: int | None = None, unsafe: bool = False
) -> np.ndarray:
    """
    Convert a list/array of base pairs to a symmetric contact map.

    Args:
        pairs: numpy array of shape (n, 2) with 0-based indices.
        length: Optional sequence length. If not provided, inferred as
            ``max_index + 1`` when pairs is non-empty; 0 otherwise.

    Returns:
        A boolean numpy array of shape (length, length).

    Raises:
        ValueError: If indices are negative or exceed the provided length.

    Examples:
        >>> pairs_to_contact_map([(0, 4), (1, 3)], length=5).astype(int)
        array([[0, 0, 0, 0, 1],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [1, 0, 0, 0, 0]])
    """
    if not isinstance(pairs, np.ndarray):
        pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        max_index = -1
    else:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be a sequence/array of (i, j) index tuples")
        pairs = pairs.astype(int, copy=False)
        low = np.minimum(pairs[:, 0], pairs[:, 1])
        high = np.maximum(pairs[:, 0], pairs[:, 1])
        pairs = np.stack([low, high], axis=1)
        max_index = np.max(pairs)

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


def contact_map_to_pairs(contact_map: np.ndarray | list | Tensor, unsafe: bool = False) -> np.ndarray:
    """
    Convert a contact map to a list of base pairs.

    The function validates symmetry and diagonal zeros unless ``unsafe=True``.
    If multiple pairings for a position exist and ``unsafe=True``, it keeps the
    first valid pairing in the upper triangle and drops the rest with a warning.

    Args:
        contact_map: Square matrix where non-zero indicates a pair.
        unsafe: Relax validations; symmetrize using upper triangle and remove
            diagonal/non-unique pairings with warnings.

    Returns:
        A numpy array of shape (n, 2) with pairs (i, j) where i < j.

    Raises:
        ValueError: On invalid shape/symmetry/diagonal/multi-pairing when not unsafe.

    Examples:
        Basic symmetric contact map
        >>> contact_map_to_pairs([
        ...   [0, 0, 0, 0, 1],
        ...   [0, 0, 0, 1, 0],
        ...   [0, 0, 0, 0, 0],
        ...   [0, 1, 0, 0, 0],
        ...   [1, 0, 0, 0, 0],
        ... ])
        array([[0, 4],
               [1, 3]])

        Non-symmetric input handled with unsafe=True
        >>> import numpy as np
        >>> cm = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        >>> contact_map_to_pairs(cm, unsafe=True)
        array([[0, 1]])
    """
    if isinstance(contact_map, list):
        contact_map = np.asarray(contact_map)
    elif isinstance(contact_map, Tensor):
        contact_map = contact_map.detach().cpu().numpy()
    contact_map = contact_map != 0

    if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
        raise ValueError("Contact map must be a square 2D matrix.")
    n = contact_map.shape[0]

    if not np.array_equal(contact_map, contact_map.T):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        warn("Contact map is not symmetric.\nUsing only the upper triangular part.")
        triu = np.triu(contact_map)
        contact_map = triu | triu.T

    if np.any(np.diag(contact_map)):
        if not unsafe:
            raise ValueError(
                "Contact map diagonal must be zero (bases cannot pair with themselves).\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn("Contact map diagonal is not zero (bases cannot pair with themselves).\nSetting diagonal to zero.")
        np.fill_diagonal(contact_map, False)

    row_sums = np.count_nonzero(contact_map, axis=1)
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

    # Fast path: valid maps with <=1 pairing per position
    if not len(multiple_pairings):
        ii, jj = np.where(np.triu(contact_map, k=1))
        return np.column_stack((ii, jj))

    # Unsafe path: build all upper-triangular edges once, then greedily accept
    # in lexicographic order (i asc, j asc) ensuring each index is used at most once.
    ii, jj = np.where(np.triu(contact_map, k=1))
    if ii.size == 0:
        return np.empty((0, 2), dtype=int)
    # sort edges by i then j for deterministic greedy resolution
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


def pairs_to_dot_bracket(pairs: np.ndarray | Sequence[Tuple[int, int]], length: int, unsafe: bool = False) -> str:
    """
    Convert a list/array of base pairs to a dot-bracket string.

    The algorithm assigns non-crossing pairs to the same bracket type tier.
    Crossing pairs are assigned to subsequent tiers to represent pseudoknots.

    Args:
        pairs: numpy array of pairs (i, j) with shape (n, 2), 0-based.
        length: Total length of the sequence.
        unsafe: If True, drop pairs that are out-of-bounds or duplicate for an
            index, and continue with warnings. If False, raise errors.

    Returns:
        Dot-bracket string of length ``length``.

    Raises:
        ValueError: On invalid indices or too many pseudoknot tiers for bracket types.

    Examples:
        Nested pairs
        >>> pairs_to_dot_bracket([(0, 3), (1, 2)], length=4)
        '(())'

        Crossing pairs (pseudoknot)
        >>> pairs_to_dot_bracket([(0, 2), (1, 3)], length=4)
        '([)]'

        Out-of-bounds index raises in strict mode
        >>> pairs_to_dot_bracket([(0, 5)], length=4)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Pair (0, 5) is out of bounds for length 4.

        Too many pseudoknot tiers exceed available bracket types
        >>> m = len(_DOT_BRACKET_PAIR_TABLE) + 1  # one more than supported tiers
        >>> pairs = [(i, i + m) for i in range(m)]  # mutually crossing tiers
        >>> pairs_to_dot_bracket(pairs, length=2*m)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not represent all base pairs with available bracket types.
    """
    # Normalize and validate in a vectorized way for arrays; fall back to array conversion otherwise
    # Convert to array and vectorize normalization + filtering
    if not isinstance(pairs, np.ndarray):
        pairs = np.asarray(list(pairs), dtype=int)

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

    # Greedy uniqueness per index in sorted order (vector-friendly, minimal loop)
    seen = np.zeros(length, dtype=bool)
    order = np.lexsort((normalized_arr[:, 1], normalized_arr[:, 0]))
    norm_sorted = normalized_arr[order]
    keep_idx = []
    dup_positions: List[int] = []
    for idx in range(norm_sorted.shape[0]):
        i, j = int(norm_sorted[idx, 0]), int(norm_sorted[idx, 1])
        if seen[i] or seen[j]:
            if unsafe:
                dup_positions.extend([i, j])
                continue
            raise ValueError(f"Positions {i} or {j} are paired multiple times; not allowed.")
        seen[i] = True
        seen[j] = True
        keep_idx.append(idx)
    if dup_positions and unsafe:
        warn(
            f"Some positions appear in multiple pairs: {sorted(set(dup_positions))}.\n"
            "Keeping only the first occurrence."
        )
    filtered_arr = norm_sorted[keep_idx] if keep_idx else np.empty((0, 2), dtype=int)

    dot_bracket = ["." for _ in range(length)]
    bracket_types = list(_DOT_BRACKET_PAIR_TABLE.items())

    # Pack pairs into the minimum number of non-crossing tiers (pseudoknot layers)
    # using a stack-of-ends per tier to check crossing in O(1) amortized.
    tiers: list[list[Tuple[int, int]]] = []
    end_stacks: list[list[int]] = []
    for i, j in filtered_arr.tolist():
        placed = False
        for tier_k, stack in zip(tiers, end_stacks):
            while stack and stack[-1] < i:
                stack.pop()
            if not stack or j < stack[-1]:
                tier_k.append((i, j))
                stack.append(j)
                placed = True
                break
        if not placed:
            tiers.append([(i, j)])
            end_stacks.append([j])

    if len(tiers) > len(bracket_types):
        if not unsafe:
            raise ValueError(
                "Could not represent all base pairs with available bracket types.\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn(
            "Could not represent all base pairs with available bracket types.\n"
            f"Omitting {sum(len(t) for t in tiers[len(bracket_types):])} pairs."
        )
        tiers = tiers[: len(bracket_types)]

    for tier, (open_bracket, close_bracket) in zip(tiers, bracket_types):
        for i, j in tier:
            dot_bracket[i] = open_bracket
            dot_bracket[j] = close_bracket

    return "".join(dot_bracket)


def dot_bracket_to_contact_map(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a contact map.

    This function parses a dot-bracket string representation of secondary structure
    and creates a binary contact map where 1 indicates paired bases.

    Note:
        This function is the inverse of
            [`contact_map_to_dot_bracket`][multimolecule.data.rna_secondary_structure.contact_map_to_dot_bracket].

    Args:
        dot_bracket: A string in dot-bracket notation representing secondary structure.
            Can include various bracket types for representing pseudoknots.

    Returns:
        A symmetric numpy array of shape (n, n) where n is the length of the input string.
            1 indicates paired bases, 0 indicates unpaired bases.

    Raises:
        ValueError: If the dot-bracket notation contains unmatched or invalid symbols.

    Examples:
        Simple hairpin structure
        >>> dot_bracket_to_contact_map("((.))").astype(int)
        array([[0, 0, 0, 0, 1],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [1, 0, 0, 0, 0]])

        Structure with pseudoknots using different bracket types
        >>> dot_bracket_to_contact_map("((.[.))]").astype(int)
        array([[0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0]])

        Unpaired sequence
        >>> dot_bracket_to_contact_map("...").astype(int)
        array([[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]])

        Error case: unmatched bracket
        >>> dot_bracket_to_contact_map("((.)")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unmatched symbol ( in dot-bracket notation
    """
    return pairs_to_contact_map(dot_bracket_to_pairs(dot_bracket), length=len(dot_bracket))


def contact_map_to_dot_bracket(contact_map: np.ndarray | list | Tensor, unsafe: bool = False) -> str:
    """
    Convert a contact map to dot-bracket notation.

    This function takes a contact map (binary matrix where 1 indicates paired bases)
    and converts it to a dot-bracket notation string.

    Note:
        This function is the inverse of
            [`dot_bracket_to_contact_map`][multimolecule.data.rna_secondary_structure.dot_bracket_to_contact_map].

    Args:
        contact_map: A square array where 1 indicates a base pair.
            Should be symmetric with zeros on the diagonal.
        unsafe: If True, only use the upper triangular part of the matrix and issue a warning
            if the matrix is not symmetric. Useful for processing model outputs.

    Returns:
        A string in dot-bracket notation representing the secondary structure.

    Raises:
        ValueError: If the contact map contains invalid pairing information or
            if there are too many pseudoknots to represent with available bracket types.

    Examples:
        Simple hairpin structure
        >>> contact_map = [
        ...     [0, 0, 0, 0, 1],
        ...     [0, 0, 0, 1, 0],
        ...     [0, 0, 0, 0, 0],
        ...     [0, 1, 0, 0, 0],
        ...     [1, 0, 0, 0, 0]
        ... ]
        >>> contact_map_to_dot_bracket(contact_map)
        '((.))'

        Structure with pseudoknots using different bracket types
        >>> import numpy as np
        >>> contact_map = np.array([
        ...     [0, 0, 0, 0, 0, 0, 1, 0],
        ...     [0, 0, 0, 0, 0, 1, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0, 1],
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 1, 0, 0, 0, 0, 0, 0],
        ...     [1, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 0, 0, 0, 0]
        ... ])
        >>> contact_map_to_dot_bracket(contact_map)
        '((.[.))]'

        Unpaired sequence
        >>> import torch
        >>> contact_map = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        >>> contact_map_to_dot_bracket(contact_map)
        '...'

        Using unsafe mode with non-symmetric input
        >>> import numpy as np
        >>> contact_map = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        >>> contact_map_to_dot_bracket(contact_map, unsafe=True)  # Will issue a warning
        '().'

        Using unsafe mode with non-symmetric input
        >>> import numpy as np
        >>> contact_map = np.array([
        ...     [0, 1, 1, 0, 0],
        ...     [1, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 1],
        ...     [0, 0, 0, 1, 0],
        ... ])
        >>> contact_map_to_dot_bracket(contact_map, unsafe=True)
        '().()'
    """
    return pairs_to_dot_bracket(
        contact_map_to_pairs(contact_map, unsafe=unsafe), length=len(contact_map), unsafe=unsafe
    )
