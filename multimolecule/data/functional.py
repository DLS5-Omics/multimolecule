# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

dot_bracket_to_contact_map_table = str.maketrans(
    {",": ".", "_": ".", "[": "(", "]": ")", "{": "(", "}": ")", "<": "(", ">": ")"}
)


def dot_bracket_to_contact_map(dot_bracket: str):
    dot_bracket = dot_bracket.translate(dot_bracket_to_contact_map_table)
    n = len(dot_bracket)
    contact_map = np.zeros((n, n), dtype=int)
    stack = []
    for i, symbol in enumerate(dot_bracket):
        if symbol == "(":
            stack.append(i)
        elif symbol == ")":
            j = stack.pop()
            contact_map[i, j] = contact_map[j, i] = 1
    return contact_map
