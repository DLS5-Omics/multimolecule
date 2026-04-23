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

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_auroc as _binary_auroc
from torchmetrics.functional.classification import binary_average_precision as _binary_auprc


def _binary_stats(preds: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")
    preds = preds.to(dtype=torch.bool)
    targets = targets.to(dtype=torch.bool)
    tp = (preds & targets).sum().to(dtype=torch.float32)
    fp = (preds & ~targets).sum().to(dtype=torch.float32)
    fn = (~preds & targets).sum().to(dtype=torch.float32)
    pred_pos = tp + fp
    target_pos = tp + fn
    return tp, fp, fn, pred_pos, target_pos


def binary_precision(preds: Tensor, targets: Tensor) -> Tensor:
    tp, _, _, pred_pos, target_pos = _binary_stats(preds, targets)
    if pred_pos.item() == 0:
        if target_pos.item() == 0:
            return tp.new_full((), float("nan"))
        return tp.new_zeros(())
    return tp / pred_pos


def binary_recall(preds: Tensor, targets: Tensor) -> Tensor:
    tp, _, _, pred_pos, target_pos = _binary_stats(preds, targets)
    if target_pos.item() == 0:
        if pred_pos.item() > 0:
            return tp.new_zeros(())
        return tp.new_full((), float("nan"))
    return tp / target_pos


def binary_f1_score(preds: Tensor, targets: Tensor) -> Tensor:
    tp, fp, fn, pred_pos, target_pos = _binary_stats(preds, targets)
    if pred_pos.item() == 0 and target_pos.item() == 0:
        return tp.new_full((), float("nan"))
    return (2 * tp) / ((2 * tp) + fp + fn)


def binary_mcc(preds: Tensor, targets: Tensor) -> Tensor:
    tp, fp, fn, pred_pos, target_pos = _binary_stats(preds, targets)
    if pred_pos.item() == 0 and target_pos.item() == 0:
        return tp.new_full((), float("nan"))
    total = torch.tensor(preds.numel(), device=tp.device, dtype=torch.float32)
    if total.item() == 0:
        return tp.new_full((), float("nan"))
    tn = total - tp - fp - fn
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    sqrt_denom = torch.sqrt(denom)
    zero = tp.new_zeros(())
    return torch.where(sqrt_denom > 0, (tp * tn - fp * fn) / sqrt_denom, zero)


def binary_prf_mcc(preds: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tp, fp, fn, pred_pos, target_pos = _binary_stats(preds, targets)
    zero = tp.new_zeros(())
    nan = tp.new_full((), float("nan"))

    if pred_pos.item() == 0 and target_pos.item() == 0:
        return nan, nan, nan, nan

    precision = tp / pred_pos if pred_pos.item() > 0 else zero
    recall = tp / target_pos if target_pos.item() > 0 else zero
    f1 = (2 * tp) / ((2 * tp) + fp + fn)

    total = torch.tensor(preds.numel(), device=tp.device, dtype=torch.float32)
    if total.item() == 0:
        mcc = nan
    else:
        tn = total - tp - fp - fn
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        sqrt_denom = torch.sqrt(denom)
        mcc = torch.where(sqrt_denom > 0, (tp * tn - fp * fn) / sqrt_denom, zero)
    return precision, recall, f1, mcc


def f1_from_pr(precision: Tensor, recall: Tensor) -> Tensor:
    denom = precision + recall
    if denom.numel() == 1:
        if torch.isnan(denom):
            return torch.tensor(float("nan"), device=precision.device, dtype=precision.dtype)
        if denom.item() == 0:
            return torch.zeros((), device=precision.device, dtype=precision.dtype)
        return (2 * precision * recall) / denom
    out = torch.zeros_like(denom)
    mask = denom != 0
    if mask.any():
        out[mask] = (2 * precision[mask] * recall[mask]) / denom[mask]
    return out


def binary_auroc(scores: Tensor, targets: Tensor) -> Tensor:
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    targets_bool = targets.to(dtype=torch.bool)
    target_pos = targets_bool.sum()
    if target_pos.item() == 0:
        return scores.new_full((), float("nan"), dtype=torch.float32)
    return _binary_auroc(scores.to(dtype=torch.float32), targets.to(dtype=torch.int64))


def binary_auprc(scores: Tensor, targets: Tensor) -> Tensor:
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    targets_bool = targets.to(dtype=torch.bool)
    target_pos = targets_bool.sum()
    if target_pos.item() == 0:
        return scores.new_full((), float("nan"), dtype=torch.float32)
    return _binary_auprc(scores.to(dtype=torch.float32), targets.to(dtype=torch.int64))
