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

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Headless rendering for all tests in this package.

NESTED_SEQUENCE = "GGGAAAUCCCAAUUGCAAUU"
NESTED_DOT_BRACKET = "(((...)))...((....))"
NESTED_REFERENCE = "((....)).....((..)).."[: len(NESTED_DOT_BRACKET)]

PSEUDOKNOT_SEQUENCE = "GGGGAAAAGGGGAAAACCCCAAAACCCC"
PSEUDOKNOT_DOT_BRACKET = "((((....[[[[....))))....]]]]"


@pytest.fixture
def nested_structure() -> tuple[str, str]:
    return NESTED_SEQUENCE, NESTED_DOT_BRACKET


@pytest.fixture
def nested_reference() -> str:
    return NESTED_REFERENCE


@pytest.fixture
def pseudoknot_structure() -> tuple[str, str]:
    return PSEUDOKNOT_SEQUENCE, PSEUDOKNOT_DOT_BRACKET


@pytest.fixture
def random_contact_map() -> np.ndarray:
    rng = np.random.default_rng(0)
    length = len(NESTED_DOT_BRACKET)
    m = rng.random((length, length))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m
