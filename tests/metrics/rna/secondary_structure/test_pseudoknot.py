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

from multimolecule.metrics.rna.secondary_structure import pseudoknot
from tests.metrics.rna.secondary_structure.utils import (
    assert_context_confusion_metric_family,
    assert_precision_recall_curve_contract,
    context_from_dot_bracket,
)


def test_crossing_metrics_match_numeric_fixture(crossing_miss_context) -> None:
    expected_pair_confusion = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    expected_nt_confusion = torch.tensor([[0, 0], [4, 0]], dtype=torch.int64)
    expected_event_confusion = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    expected_stem_confusion = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    device = crossing_miss_context.device

    assert torch.allclose(
        pseudoknot.crossing_pairs_confusion(crossing_miss_context), expected_pair_confusion.to(device)
    )
    assert torch.equal(
        pseudoknot.crossing_nucleotides_confusion(crossing_miss_context), expected_nt_confusion.to(device)
    )
    assert torch.allclose(crossing_miss_context.crossing_events_confusion, expected_event_confusion.to(device))
    assert torch.allclose(crossing_miss_context.crossing_stem_confusion, expected_stem_confusion.to(device))

    assert pseudoknot.crossing_pairs_precision(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_pairs_recall(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_pairs_f1(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_nucleotides_precision(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_nucleotides_recall(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_nucleotides_f1(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_events_precision(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_events_recall(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_events_f1(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_stem_precision(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_stem_recall(crossing_miss_context) == pytest.approx(0.0)
    assert pseudoknot.crossing_stem_f1(crossing_miss_context) == pytest.approx(0.0)


def test_crossing_event_metric_family_uses_context_confusion(perfect_crossing_context) -> None:
    assert_context_confusion_metric_family(
        perfect_crossing_context,
        perfect_crossing_context.crossing_events_confusion,
        pseudoknot.crossing_events_precision,
        pseudoknot.crossing_events_recall,
        pseudoknot.crossing_events_f1,
    )


@pytest.mark.parametrize(
    "metric",
    [
        pseudoknot.crossing_events_precision_recall_curve,
        pseudoknot.crossing_nucleotides_precision_recall_curve,
        pseudoknot.crossing_pairs_precision_recall_curve,
        pseudoknot.crossing_stem_precision_recall_curve,
    ],
)
@pytest.mark.parametrize(
    ("dot_bracket", "sequence"),
    [
        pytest.param("(([[))]]", "ACGAUGUC", id="positive"),
        pytest.param("....", "AAAA", id="empty"),
    ],
)
def test_crossing_precision_recall_curve_contract(metric, dot_bracket: str, sequence: str) -> None:
    context = context_from_dot_bracket(dot_bracket, sequence=sequence)
    assert_precision_recall_curve_contract(metric(context))
