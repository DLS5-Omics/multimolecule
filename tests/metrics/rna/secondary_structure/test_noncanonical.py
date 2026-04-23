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
import torch

from multimolecule.metrics.rna.secondary_structure import noncanonical
from tests.metrics.rna.secondary_structure.utils import assert_precision_recall_curve_contract, context_from_dot_bracket


def test_noncanonical_pair_metrics_match_numeric_fixture(noncanonical_context) -> None:
    expected_confusion = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32, device=noncanonical_context.device)
    assert torch.allclose(noncanonical.noncanonical_pairs_confusion(noncanonical_context), expected_confusion)
    assert noncanonical.noncanonical_pairs_precision(noncanonical_context) == pytest.approx(0.5)
    assert noncanonical.noncanonical_pairs_recall(noncanonical_context) == pytest.approx(1.0)
    assert noncanonical.noncanonical_pairs_f1(noncanonical_context) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("dot_bracket", "sequence"),
    [
        pytest.param("(([[))]]", "ACGAUGUC", id="positive"),
        pytest.param("....", "AAAA", id="empty"),
    ],
)
def test_noncanonical_precision_recall_curve_contract(dot_bracket: str, sequence: str) -> None:
    context = context_from_dot_bracket(dot_bracket, sequence=sequence)
    assert_precision_recall_curve_contract(noncanonical.noncanonical_pairs_precision_recall_curve(context))
