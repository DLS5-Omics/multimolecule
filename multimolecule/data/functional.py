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
from warnings import warn

import numpy as np
from torch import Tensor

dot_bracket_pair_table = {"(": ")", "[": "]", "{": "}", "<": ">"}
dot_bracket_pair_table.update(zip(string.ascii_uppercase, string.ascii_lowercase))
reverse_dot_bracket_pair_table = {v: k for k, v in dot_bracket_pair_table.items()}


def dot_bracket_to_contact_map(dot_bracket: str) -> np.ndarray:
    """
    Convert a dot-bracket notation string to a contact map.

    This function parses a dot-bracket string representation of secondary structure
    and creates a binary contact map where 1 indicates paired bases.

    Note:
        This function is the inverse of
            [`contact_map_to_dot_bracket`][multimolecule.data.functional.contact_map_to_dot_bracket].

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
        >>> dot_bracket_to_contact_map("((.))")
        array([[0, 0, 0, 0, 1],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [1, 0, 0, 0, 0]])

        Structure with pseudoknots using different bracket types
        >>> dot_bracket_to_contact_map("((.[.))]")
        array([[0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0]])

        Unpaired sequence
        >>> dot_bracket_to_contact_map("...")
        array([[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]])

        Error case: unmatched bracket
        >>> dot_bracket_to_contact_map("((.)")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unmatched symbol ( in dot-bracket notation
    """
    n = len(dot_bracket)
    contact_map = np.zeros((n, n), dtype=int)

    stacks: defaultdict[str, list[int]] = defaultdict(list)
    for i, symbol in enumerate(dot_bracket):
        if symbol in dot_bracket_pair_table:
            stacks[symbol].append(i)
        elif symbol in reverse_dot_bracket_pair_table:
            try:
                j = stacks[reverse_dot_bracket_pair_table[symbol]].pop()
            except IndexError:
                raise ValueError(f"Unmatched symbol {symbol} at position {i} in sequence {dot_bracket}")
            contact_map[i, j] = contact_map[j, i] = 1
        elif symbol not in {".", ",", "_"}:
            raise ValueError(f"Invalid symbol {symbol} at position {i} in sequence {dot_bracket}")
    for symbol, stack in stacks.items():
        if stack:
            raise ValueError(f"Unmatched symbol {symbol} at position {stack[0]} in sequence {dot_bracket}")
    return contact_map


def contact_map_to_dot_bracket(contact_map: np.ndarray | list | Tensor, unsafe: bool = False) -> str:
    """
    Convert a contact map to dot-bracket notation.

    This function takes a contact map (binary matrix where 1 indicates paired bases)
    and converts it to a dot-bracket notation string.

    Note:
        This function is the inverse of
            [`dot_bracket_to_contact_map`][multimolecule.data.functional.dot_bracket_to_contact_map].

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
        >>> contact_map = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        >>> contact_map_to_dot_bracket(contact_map, unsafe=True)  # Will issue a warning
        '().'
    """
    if isinstance(contact_map, list):
        contact_map = np.array(contact_map)
    elif isinstance(contact_map, Tensor):
        contact_map = contact_map.detach().cpu().numpy()
    contact_map = contact_map.astype(int)
    n = contact_map.shape[0]

    if not np.array_equal(contact_map, contact_map.T):
        if not unsafe:
            raise ValueError("Contact map is not symmetric.\nPass `unsafe=True` if this is expected.")
        warn("Contact map is not symmetric.\nUsing only the upper triangular part.")
        triu = np.triu(contact_map)
        contact_map = triu + triu.T

    if not np.all(np.diag(contact_map) == 0):
        if not unsafe:
            raise ValueError(
                "Contact map diagonal must be zero (bases cannot pair with themselves).\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn("Contact map diagonal is not zero (bases cannot pair with themselves).\nSetting diagonal to zero.")
        np.fill_diagonal(contact_map, 0)

    dot_bracket = ["." for _ in range(n)]
    bracket_types = list(dot_bracket_pair_table.items())
    remaining_contacts = contact_map.copy()

    for open_bracket, close_bracket in bracket_types:
        if np.count_nonzero(remaining_contacts) == 0:
            break
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if remaining_contacts[i, j] == 1:
                    crosses_existing = False
                    for start, end in pairs:
                        if start < i < end < j or i < start < j < end:
                            crosses_existing = True
                            break
                    if not crosses_existing:
                        pairs.append((i, j))
        if not pairs:
            continue
        for i, j in pairs:
            dot_bracket[i] = open_bracket
            dot_bracket[j] = close_bracket
            remaining_contacts[i, j] = remaining_contacts[j, i] = 0

    if np.count_nonzero(remaining_contacts) > 0:
        if not unsafe:
            raise ValueError(
                "Could not represent all base pairs in the contact map with available bracket types.\n"
                "Pass `unsafe=True` if this is expected."
            )
        warn(
            "Could not represent all base pairs in the contact map with available bracket types.\n"
            f"Omitting {np.count_nonzero(remaining_contacts) // 2} pairs."
        )

    return "".join(dot_bracket)
