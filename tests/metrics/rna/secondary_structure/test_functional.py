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

from multimolecule.metrics.rna.secondary_structure import functional


def test_binary_stats_and_score_shape_checks() -> None:
    with pytest.raises(ValueError, match="same shape"):
        functional._binary_stats(torch.ones((2, 2)), torch.ones((2, 3)))

    with pytest.raises(ValueError, match="same shape"):
        functional.binary_auroc(torch.tensor([0.1, 0.2]), torch.tensor([1]))

    with pytest.raises(ValueError, match="same shape"):
        functional.binary_auprc(torch.tensor([0.1, 0.2]), torch.tensor([1]))


def test_binary_mcc_empty_and_f1_vectorized_mask_branch() -> None:
    empty = torch.empty((0,), dtype=torch.int64)
    mcc = functional.binary_mcc(empty, empty)
    assert math.isnan(float(mcc.item()))

    precision = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    recall = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    f1 = functional.f1_from_pr(precision, recall)
    assert torch.allclose(f1, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))


def test_binary_recall_semantics_when_no_positive_targets() -> None:
    # pred_pos > 0, target_pos == 0: recall = 0 (model predicted wrong things)
    preds = torch.tensor([1, 1, 0], dtype=torch.int64)
    targets = torch.zeros((3,), dtype=torch.int64)

    recall = functional.binary_recall(preds, targets)
    assert float(recall.item()) == 0.0

    p, r, f1, mcc = functional.binary_prf_mcc(preds, targets)
    assert float(p.item()) == 0.0
    assert float(r.item()) == 0.0
    assert float(f1.item()) == 0.0

    # pred_pos == 0, target_pos == 0: recall = NaN (no such structure, skip)
    empty_preds = torch.zeros((3,), dtype=torch.int64)
    recall_nan = functional.binary_recall(empty_preds, targets)
    assert math.isnan(float(recall_nan.item()))


def test_binary_auroc_auprc_nan_when_no_positive_targets() -> None:
    scores = torch.tensor([0.9, 0.1, 0.2], dtype=torch.float32)
    targets = torch.zeros((3,), dtype=torch.int64)

    auroc = functional.binary_auroc(scores, targets)
    auprc = functional.binary_auprc(scores, targets)

    assert math.isnan(float(auroc.item()))
    assert math.isnan(float(auprc.item()))
