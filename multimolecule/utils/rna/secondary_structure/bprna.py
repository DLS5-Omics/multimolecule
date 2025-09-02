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

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from .topology import RnaSecondaryStructure

_STRUCT_LABEL_MAP = np.array(["", "H", "B", "I", "M", "X", "E"], dtype="<U1")


def annotate_structure(structure: RnaSecondaryStructure) -> str:
    """
    Return a bpRNA-like structural annotation string (structure array) for this structure.

    Labels: S (stems), H (hairpins), B (bulges), I (internals), M (multibranch),
    X (external), E (end).
    """
    n = len(structure.sequence)
    if n == 0:
        return ""

    arr = np.full(n, "E", dtype="<U1")

    primary_pairs = structure.primary_pairs
    if len(primary_pairs):
        primary_np = primary_pairs.detach().cpu().numpy()
        arr[primary_np[:, 0]] = "S"
        arr[primary_np[:, 1]] = "S"

    labels_np = structure.loop_labels.detach().cpu().numpy()
    if labels_np.size:
        mask = (labels_np > 0) & (arr != "S")
        arr[mask] = _STRUCT_LABEL_MAP[labels_np[mask]]

    return "".join(arr.tolist())


def annotate_function(structure: RnaSecondaryStructure) -> str:
    """
    Return a bpRNA-like functional annotation string (knot/function array) for this structure.

    Labels: K for bases involved in pseudoknot pairs, N otherwise.
    """
    n = len(structure.sequence)
    arr = ["N"] * n
    for i, j in structure.pseudoknot_pairs.tolist():
        if 0 <= i < n:
            arr[i] = "K"
        if 0 <= j < n:
            arr[j] = "K"
    return "".join(arr)


def annotate(structure: RnaSecondaryStructure) -> Tuple[str, str]:
    """
    Return both structural and functional annotations for this structure.
    """
    return annotate_structure(structure), annotate_function(structure)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from multimolecule.io import load, write_st

    path = Path(sys.argv[1])

    write_st(load(path), path.with_suffix(".st"))
