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

from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor

from .common import pair_error_rate as pair_error_rate_from_pairs
from .common import pair_exact_match as pair_exact_match_from_pairs
from .functional import binary_f1_score
from .functional import binary_mcc as binary_mcc_from_labels
from .functional import binary_precision as binary_precision_from_labels
from .functional import binary_recall as binary_recall_from_labels

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def _pair_labels(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor]:
    tri = torch.triu_indices(context.length, context.length, offset=1, device=context.device)
    preds = context.pred[tri[0], tri[1]]
    targets = context.target_pairs.new_zeros((tri.shape[1],), dtype=torch.bool)
    if context.target_pairs.numel() != 0:
        pair_ids = context.target_pairs[:, 0] * context.length + context.target_pairs[:, 1]
        tri_ids = tri[0] * context.length + tri[1]
        targets = torch.isin(tri_ids, pair_ids)
    if preds.is_floating_point():
        preds = preds >= context.threshold
    return preds.to(dtype=torch.int64), targets.to(dtype=torch.int64)


def binary_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _pair_labels(context)
    return binary_f1_score(preds, targets)


def binary_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _pair_labels(context)
    return binary_mcc_from_labels(preds, targets)


def binary_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _pair_labels(context)
    return binary_precision_from_labels(preds, targets)


def binary_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _pair_labels(context)
    return binary_recall_from_labels(preds, targets)


def pair_exact_match(context: RnaSecondaryStructureContext) -> Tensor:
    return pair_exact_match_from_pairs(context.pred_pairs, context.target_pairs, context.device)


def pair_error_rate(context: RnaSecondaryStructureContext) -> Tensor:
    return pair_error_rate_from_pairs(context.pred_pairs, context.target_pairs, context.device)
