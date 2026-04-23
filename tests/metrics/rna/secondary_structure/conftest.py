# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
#
# This file is part of MultiMolecule.
#
# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

import pytest

from multimolecule.metrics.rna.secondary_structure import RnaSecondaryStructureContext
from tests.metrics.rna.secondary_structure.utils import context_from_dot_bracket, context_from_pred_target


@pytest.fixture(scope="module")
def perfect_nested_context() -> RnaSecondaryStructureContext:
    return context_from_dot_bracket("((..))", sequence="GCAUCG")


@pytest.fixture(scope="module")
def perfect_crossing_context() -> RnaSecondaryStructureContext:
    return context_from_dot_bracket("([)]", sequence="AUGC")


@pytest.fixture(scope="module")
def loop_overlap_context() -> RnaSecondaryStructureContext:
    return context_from_pred_target("((....))", "((..))..", sequence="AAAAAAAA")


@pytest.fixture(scope="module")
def noncanonical_context() -> RnaSecondaryStructureContext:
    return context_from_pred_target(".(()).", "((..))", sequence="ACGAUU")


@pytest.fixture(scope="module")
def crossing_miss_context() -> RnaSecondaryStructureContext:
    return context_from_pred_target("(())", "([)]", sequence="AUGC")


@pytest.fixture(scope="module")
def pr_positive_context() -> RnaSecondaryStructureContext:
    return context_from_dot_bracket("(([[))]]", sequence="ACGAUGUC")


@pytest.fixture(scope="module")
def pr_empty_context() -> RnaSecondaryStructureContext:
    return context_from_dot_bracket("....", sequence="AAAA")
