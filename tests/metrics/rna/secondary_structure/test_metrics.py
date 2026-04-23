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

import math

import pytest
import torch
from danling import NestedTensor

from multimolecule.metrics.rna.secondary_structure import RnaSecondaryStructureMetrics
from multimolecule.metrics.rna.secondary_structure import metrics as metrics_mod
from multimolecule.utils.rna.secondary_structure import pairs_to_contact_map


def _contact_map(pairs: list[tuple[int, int]], length: int) -> torch.Tensor:
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    return pairs_to_contact_map(pair_tensor, length=length)


def _metric_value(metrics: RnaSecondaryStructureMetrics, name: str) -> float:
    return metrics[name].avg


def _assert_metric(metrics: RnaSecondaryStructureMetrics, name: str, expected: float) -> None:
    value = _metric_value(metrics, name)
    if math.isnan(expected):
        assert math.isnan(value), f"{name} expected NaN, got {value}"
    else:
        assert value == pytest.approx(expected), f"{name} expected {expected}, got {value}"


_PAIR_TRUTH_TABLE = [
    pytest.param(
        "AUAU",
        [(0, 3), (1, 2)],
        [(0, 3), (1, 2)],
        {
            "binary_precision": 1.0,
            "binary_recall": 1.0,
            "binary_f1": 1.0,
            "pair_exact_match": 1.0,
            "pair_error_rate": 0.0,
            "paired_nucleotides_precision": 1.0,
            "paired_nucleotides_recall": 1.0,
            "paired_nucleotides_f1": 1.0,
        },
        id="perfect",
    ),
    pytest.param(
        "AUAU",
        [(0, 3)],
        [(0, 3), (1, 2)],
        {
            "binary_precision": 1.0,
            "binary_recall": 0.5,
            "binary_f1": 2.0 / 3.0,
            "pair_exact_match": 0.0,
            "pair_error_rate": 0.5,
            "paired_nucleotides_precision": 1.0,
            "paired_nucleotides_recall": 0.5,
            "paired_nucleotides_f1": 2.0 / 3.0,
        },
        id="missing-pair",
    ),
    pytest.param(
        "AUAUAU",
        [(0, 5), (1, 4)],
        [(0, 5)],
        {
            "binary_precision": 0.5,
            "binary_recall": 1.0,
            "binary_f1": 2.0 / 3.0,
            "pair_exact_match": 0.0,
            "pair_error_rate": 0.5,
            "paired_nucleotides_precision": 0.5,
            "paired_nucleotides_recall": 1.0,
            "paired_nucleotides_f1": 2.0 / 3.0,
        },
        id="extra-pair",
    ),
    pytest.param(
        "AUAUAU",
        [(0, 5)],
        [(1, 4)],
        {
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "pair_exact_match": 0.0,
            "pair_error_rate": 1.0,
            "paired_nucleotides_precision": 0.0,
            "paired_nucleotides_recall": 0.0,
            "paired_nucleotides_f1": 0.0,
        },
        id="disjoint-pairs",
    ),
    pytest.param(
        "AUAU",
        [],
        [(0, 3)],
        {
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "pair_exact_match": 0.0,
            "pair_error_rate": 1.0,
            "paired_nucleotides_precision": 0.0,
            "paired_nucleotides_recall": 0.0,
            "paired_nucleotides_f1": 0.0,
        },
        id="empty-prediction",
    ),
    pytest.param(
        "AUAU",
        [],
        [],
        {
            "binary_precision": float("nan"),
            "binary_recall": float("nan"),
            "binary_f1": float("nan"),
            "pair_exact_match": float("nan"),
            "pair_error_rate": float("nan"),
            "paired_nucleotides_precision": float("nan"),
            "paired_nucleotides_recall": float("nan"),
            "paired_nucleotides_f1": float("nan"),
        },
        id="empty-both",
    ),
]


@pytest.mark.parametrize("sequence,pred_pairs,target_pairs,expected", _PAIR_TRUTH_TABLE)
def test_metrics_pair_truth_table(
    sequence: str, pred_pairs: list[tuple[int, int]], target_pairs: list[tuple[int, int]], expected: dict[str, float]
) -> None:
    pred = _contact_map(pred_pairs, length=len(sequence))
    target = _contact_map(target_pairs, length=len(sequence))
    metrics = RnaSecondaryStructureMetrics()
    metrics.update(pred, target, sequences=sequence)
    for metric_name, metric_expected in expected.items():
        _assert_metric(metrics, metric_name, metric_expected)


def test_metrics_require_sequences() -> None:
    contact = _contact_map([(0, 3)], length=4)
    metrics = RnaSecondaryStructureMetrics()
    with pytest.raises(ValueError, match="sequences"):
        metrics.update(contact, contact)


def test_metrics_threshold_applied_to_float_scores() -> None:
    sequence = "AAAA"
    target = _contact_map([(0, 3)], length=len(sequence))
    pred = target.to(dtype=torch.float32)
    pred[0, 3] = 0.4
    pred[3, 0] = 0.4

    metrics = RnaSecondaryStructureMetrics()
    metrics.update(pred, target, sequences=sequence, threshold=0.5)

    assert _metric_value(metrics, "binary_precision") == pytest.approx(0.0)
    assert _metric_value(metrics, "binary_recall") == pytest.approx(0.0)
    assert _metric_value(metrics, "binary_f1") == pytest.approx(0.0)


def test_as_tensor_list_variants_and_errors() -> None:
    t2 = torch.zeros((4, 4), dtype=torch.float32)
    t3 = torch.zeros((2, 4, 4), dtype=torch.float32)
    t4 = torch.zeros((2, 4, 4, 1), dtype=torch.float32)
    nested = NestedTensor([t2, t2.clone()])

    out_nested = metrics_mod._as_tensor_list(nested, name="preds")
    assert len(out_nested) == 2

    out_2d = metrics_mod._as_tensor_list(t2, name="preds")
    assert len(out_2d) == 1

    out_3d = metrics_mod._as_tensor_list(t3, name="preds")
    assert len(out_3d) == 2

    out_4d = metrics_mod._as_tensor_list(t4, name="preds")
    assert len(out_4d) == 2
    assert out_4d[0].shape == (4, 4)

    out_seq = metrics_mod._as_tensor_list([t2, [[0, 1], [1, 0]]], name="preds")
    assert len(out_seq) == 2
    assert isinstance(out_seq[1], torch.Tensor)

    with pytest.raises(ValueError, match="2D, 3D"):
        metrics_mod._as_tensor_list(torch.zeros((4,)), name="preds")

    with pytest.raises(TypeError, match="Tensor, NestedTensor"):
        metrics_mod._as_tensor_list(123, name="preds")  # type: ignore[arg-type]


def test_as_sequence_list_and_batch_validations() -> None:
    assert metrics_mod._as_sequence_list("ACGU", expected=1) == ["ACGU"]
    assert metrics_mod._as_sequence_list(["AC", "GU"], expected=2) == ["AC", "GU"]

    with pytest.raises(TypeError, match="string or sequence"):
        metrics_mod._as_sequence_list(123, expected=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match batch size"):
        metrics_mod._as_sequence_list(["AC"], expected=2)

    pred = torch.zeros((2, 4, 4), dtype=torch.int64)
    target = torch.zeros((1, 4, 4), dtype=torch.int64)
    meter = RnaSecondaryStructureMetrics()
    with pytest.raises(ValueError, match="same batch size"):
        meter.update(pred, target, sequences=["ACGU", "UGCA"])


def test_update_sigmoid_for_logits_and_non_tensor_meter_values() -> None:
    sequence = "AAAA"
    target = _contact_map([(0, 3)], length=len(sequence))
    pred = torch.full((len(sequence), len(sequence)), -10.0, dtype=torch.float32)
    pred[0, 3] = pred[3, 0] = -1.0

    meter = RnaSecondaryStructureMetrics(threshold=0.1)
    with pytest.warns(UserWarning, match="Applying sigmoid"):
        meter.update(pred, target, sequences=sequence)
    assert _metric_value(meter, "binary_precision") == pytest.approx(1.0)
    assert _metric_value(meter, "binary_recall") == pytest.approx(1.0)
