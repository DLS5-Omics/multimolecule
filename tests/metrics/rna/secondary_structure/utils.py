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

from collections.abc import Callable

import pytest
import torch

from multimolecule.metrics.rna.secondary_structure import RnaSecondaryStructureContext
from multimolecule.metrics.rna.secondary_structure.common import (
    f1_from_confusion,
    precision_from_confusion,
    recall_from_confusion,
)
from multimolecule.utils.rna.secondary_structure import dot_bracket_to_contact_map


def context_from_dot_bracket(dot_bracket: str, sequence: str | None = None) -> RnaSecondaryStructureContext:
    if sequence is None:
        sequence = "A" * len(dot_bracket)
    contact_map = torch.as_tensor(dot_bracket_to_contact_map(dot_bracket), dtype=torch.bool)
    return RnaSecondaryStructureContext(contact_map, contact_map, sequence)


def context_from_pred_target(
    pred_dot_bracket: str, target_dot_bracket: str, sequence: str
) -> RnaSecondaryStructureContext:
    pred_contact_map = torch.as_tensor(dot_bracket_to_contact_map(pred_dot_bracket), dtype=torch.bool)
    target_contact_map = torch.as_tensor(dot_bracket_to_contact_map(target_dot_bracket), dtype=torch.bool)
    return RnaSecondaryStructureContext(pred_contact_map, target_contact_map, sequence)


def assert_confusion_metric_family(
    context: RnaSecondaryStructureContext,
    confusion: Callable[[RnaSecondaryStructureContext], torch.Tensor],
    precision: Callable[[RnaSecondaryStructureContext], torch.Tensor],
    recall: Callable[[RnaSecondaryStructureContext], torch.Tensor],
    f1: Callable[[RnaSecondaryStructureContext], torch.Tensor],
) -> None:
    cm = confusion(context)
    expected_precision = precision_from_confusion(cm.to(dtype=torch.float32), context.device)
    expected_recall = recall_from_confusion(cm.to(dtype=torch.float32), context.device)
    expected_f1 = f1_from_confusion(cm.to(dtype=torch.float32), context.device)
    assert torch.allclose(precision(context), expected_precision, equal_nan=True)
    assert torch.allclose(recall(context), expected_recall, equal_nan=True)
    assert torch.allclose(f1(context), expected_f1, equal_nan=True)


def assert_context_confusion_metric_family(
    context: RnaSecondaryStructureContext,
    confusion: torch.Tensor,
    precision: Callable[[RnaSecondaryStructureContext], torch.Tensor],
    recall: Callable[[RnaSecondaryStructureContext], torch.Tensor],
    f1: Callable[[RnaSecondaryStructureContext], torch.Tensor],
) -> None:
    expected_precision = precision_from_confusion(confusion, context.device)
    expected_recall = recall_from_confusion(confusion, context.device)
    expected_f1 = f1_from_confusion(confusion, context.device)
    assert torch.allclose(precision(context), expected_precision, equal_nan=True)
    assert torch.allclose(recall(context), expected_recall, equal_nan=True)
    assert torch.allclose(f1(context), expected_f1, equal_nan=True)


def assert_precision_recall_curve_contract(result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    precision, recall, thresholds = result
    assert precision.ndim == 1
    assert recall.ndim == 1
    assert thresholds.ndim == 1
    assert precision.numel() == recall.numel()
    assert precision.numel() == thresholds.numel() + 1
    assert bool(torch.isfinite(precision).all().item())
    assert bool(torch.isfinite(recall).all().item())
    assert bool(((precision >= 0) & (precision <= 1)).all().item())
    assert bool(((recall >= 0) & (recall <= 1)).all().item())
    assert precision[-1] == pytest.approx(1.0)
    assert recall[-1] == pytest.approx(0.0)
    if thresholds.numel() > 1:
        assert bool((thresholds[1:] > thresholds[:-1]).all().item())
    if recall.numel() > 1:
        assert bool((recall[:-1] >= recall[1:]).all().item())
