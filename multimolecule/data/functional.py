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

import numpy as np

dot_bracket_pair_table = {"(": ")", "[": "]", "{": "}", "<": ">"}
dot_bracket_pair_table.update(zip(string.ascii_uppercase, string.ascii_lowercase))
reverse_dot_bracket_pair_table = {v: k for k, v in dot_bracket_pair_table.items()}


def dot_bracket_to_contact_map(dot_bracket: str):
    n = len(dot_bracket)
    contact_map = np.zeros((n, n), dtype=int)

    stacks: defaultdict[str, list[int]] = defaultdict(list)
    for i, symbol in enumerate(dot_bracket):
        if symbol in dot_bracket_pair_table:
            stacks[symbol].append(i)
        elif symbol in reverse_dot_bracket_pair_table:
            j = stacks[reverse_dot_bracket_pair_table[symbol]].pop()
            contact_map[i, j] = contact_map[j, i] = 1
        elif symbol not in {".", ",", "_"}:
            raise ValueError(f"Invalid symbol {symbol} in dot-bracket notation")
    return contact_map
