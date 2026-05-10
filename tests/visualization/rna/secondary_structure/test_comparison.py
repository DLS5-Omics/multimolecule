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

import pytest

from multimolecule.visualization.rna import compare_secondary_structures


def test_identical_structures_pair_classification() -> None:
    result = compare_secondary_structures("(((...)))", "(((...)))")
    assert result["true_positives"] == 3
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["true_positive_pairs"] == [(0, 8), (1, 7), (2, 6)]


def test_partial_overlap() -> None:
    # predicted has pairs {(0,8),(1,7),(2,6)}; reference has {(0,8),(1,7)}.
    result = compare_secondary_structures("(((...)))", "((.....))")
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["false_positive_pairs"] == [(2, 6)]


def test_unpaired_structures_classify_as_all_zeros() -> None:
    result = compare_secondary_structures(".....", ".....")
    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        compare_secondary_structures("(((...)))", "(((...))")
