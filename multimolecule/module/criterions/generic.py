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
    r"""
    A generic criterion that adapts to different problem types.

    This criterion serves as a fallback option when a specific criterion is not specified.
    It can handle different problem types (regression, binary, multiclass, multilabel) and
    automatically selects the appropriate loss function based on the data characteristics
    or explicit problem_type setting.

    Attributes:
        problem_types (list): List of supported problem types.
        problem_type (str): The type of problem being solved.
        num_labels (int): Number of labels/classes in the task.

    Examples:

        Regression:
        >>> import torch
        >>> from ..heads.config import HeadConfig
        >>> criterion = Criterion(HeadConfig(num_labels=1))
        >>> input = torch.tensor([0.5, 1.2, 3.4])
        >>> target = torch.tensor([0.7, 1.0, 3.5])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.0300)
        >>> criterion.problem_type
        'regression'

        Multi-class classification:
        >>> criterion = Criterion(HeadConfig(num_labels=3))
        >>> input = torch.tensor([[0.1, 0.2, 0.9], [1.1, 0.1, 0.2], [0.2, 2.0, 0.3]])
        >>> target = torch.tensor([2, 0, 1])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.5126)
        >>> criterion.problem_type
        'multiclass'

        Multi-label classification:
        >>> criterion = Criterion(HeadConfig(num_labels=2))
        >>> input = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        >>> target = torch.tensor([[1, 0], [1, 0]])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.7637)
        >>> criterion.problem_type
        'multilabel'
    """

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
            labels = labels.to(logits.dtype)
            if self.num_labels == 1:
                return F.mse_loss(logits.squeeze(), labels.squeeze())
            logits, labels = logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            return sum(F.mse_loss(logits[:, i], labels[:, i]).sqrt() for i in range(self.num_labels))  # type: ignore
        if self.problem_type == "multiclass":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        if self.problem_type == "binary":
            if logits.ndim == labels.ndim + 1:
                logits = logits.squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
        if self.problem_type == "multilabel":
            return F.multilabel_soft_margin_loss(logits, labels.to(logits.dtype))
        raise ValueError(f"problem_type should be one of {self.problem_types}, but got {self.problem_type}")
