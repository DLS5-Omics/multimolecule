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

from typing import TYPE_CHECKING, Tuple

from torch import Tensor

from .common import (
    confusion_from_items,
    f1_from_confusion,
    pairs_precision_recall_curve,
    precision_from_confusion,
    recall_from_confusion,
)

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def noncanonical_pairs_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_nc = context.pred_noncanonical_pairs
    target_nc = context.target_noncanonical_pairs
    return confusion_from_items(pred_nc, target_nc, device)


def noncanonical_pairs_f1(context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_nc = context.pred_noncanonical_pairs
    target_nc = context.target_noncanonical_pairs
    cm = confusion_from_items(pred_nc, target_nc, device)
    return f1_from_confusion(cm, device)


def noncanonical_pairs_precision(context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_nc = context.pred_noncanonical_pairs
    target_nc = context.target_noncanonical_pairs
    cm = confusion_from_items(pred_nc, target_nc, device)
    return precision_from_confusion(cm, device)


def noncanonical_pairs_recall(context: RnaSecondaryStructureContext) -> Tensor:
    device = context.device
    pred_nc = context.pred_noncanonical_pairs
    target_nc = context.target_noncanonical_pairs
    cm = confusion_from_items(pred_nc, target_nc, device)
    return recall_from_confusion(cm, device)


def noncanonical_pairs_precision_recall_curve(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor, Tensor]:
    return pairs_precision_recall_curve(context.pred, context.target_noncanonical_pairs)
