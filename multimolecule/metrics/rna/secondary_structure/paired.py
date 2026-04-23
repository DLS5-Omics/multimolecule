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
from torchmetrics.functional.classification import binary_confusion_matrix

from .functional import binary_f1_score, binary_mcc, binary_precision, binary_recall

if TYPE_CHECKING:
    from .context import RnaSecondaryStructureContext


def _paired_nucleotide_labels(context: RnaSecondaryStructureContext) -> Tuple[Tensor, Tensor]:
    return context.paired_labels


def paired_nucleotides_confusion(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _paired_nucleotide_labels(context)
    return binary_confusion_matrix(preds, targets)


def paired_nucleotides_f1(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _paired_nucleotide_labels(context)
    return binary_f1_score(preds, targets)


def paired_nucleotides_mcc(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _paired_nucleotide_labels(context)
    return binary_mcc(preds, targets)


def paired_nucleotides_precision(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _paired_nucleotide_labels(context)
    return binary_precision(preds, targets)


def paired_nucleotides_recall(context: RnaSecondaryStructureContext) -> Tensor:
    preds, targets = _paired_nucleotide_labels(context)
    return binary_recall(preds, targets)
