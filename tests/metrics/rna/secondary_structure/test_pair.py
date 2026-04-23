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

from multimolecule.metrics.rna.secondary_structure import pair


@pytest.mark.parametrize(
    ("metric", "lower", "upper"),
    [
        pytest.param(pair.binary_precision, 0.0, 1.0, id="precision"),
        pytest.param(pair.binary_recall, 0.0, 1.0, id="recall"),
        pytest.param(pair.binary_f1, 0.0, 1.0, id="f1"),
        pytest.param(pair.binary_mcc, -1.0, 1.0, id="mcc"),
    ],
)
def test_binary_pair_metrics_return_bounded_scalars(perfect_nested_context, metric, lower: float, upper: float) -> None:
    value = metric(perfect_nested_context)
    assert value.ndim == 0
    assert lower <= float(value.item()) <= upper


def test_pair_exact_match_and_error_rate_are_scalars(perfect_nested_context) -> None:
    assert pair.pair_exact_match(perfect_nested_context).ndim == 0
    error_rate = pair.pair_error_rate(perfect_nested_context)
    assert error_rate.ndim == 0
    assert float(error_rate.item()) >= 0.0
