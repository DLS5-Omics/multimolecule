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

from typing import TYPE_CHECKING
from warnings import warn

from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from ..heads.config import HeadConfig

from .registry import CRITERIONS


@CRITERIONS.register(default=True)
class Criterion(nn.Module):

    problem_types = ["regression", "binary", "multiclass", "multilabel"]

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.problem_type = config.problem_type
        self.num_labels = config.num_labels

    def forward(self, logits: Tensor | NestedTensor, labels: Tensor | NestedTensor) -> Tensor | None:
        if labels is None:
            return None
        if self.problem_type is None:
            if labels.is_floating_point():
                self.problem_type = "regression"
            elif self.num_labels == 1:
                self.problem_type = "binary"
            elif labels.unique().numel() == 2:
                self.problem_type = "multilabel"
            else:
                self.problem_type = "multiclass"
            warn(
                f"`problem_type` is not set. Assuming {self.problem_type}. \n"
                "This can lead to unexpected behavior. Please set `problem_type` explicitly."
            )
            self.config.problem_type = self.problem_type
        if self.problem_type == "regression":
            if logits.ndim == labels.ndim + 1:
                labels = labels.unsqueeze(-1)
            return F.mse_loss(logits, labels.to(logits.dtype))
        if self.problem_type == "multiclass":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        if self.problem_type == "binary":
            if logits.ndim == labels.ndim + 1:
                logits = logits.squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
        if self.problem_type == "multilabel":
            if labels.ndim > 2:
                logits, labels = logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1))
            return F.multilabel_soft_margin_loss(logits, labels.to(logits.dtype))
        raise ValueError(f"problem_type should be one of {self.problem_types}, but got {self.problem_type}")
