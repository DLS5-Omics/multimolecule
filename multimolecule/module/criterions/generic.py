# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F

from multimolecule.models.configuration_utils import HeadConfig


class Criterion(nn.Module):

    problem_types = ["regression", "single_label_classification", "multi_label_classification"]

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.problem_type = config.problem_type
        self.num_labels = config.num_labels

    def forward(self, logits: Tensor | NestedTensor, labels: Tensor | NestedTensor) -> Tensor | None:
        if labels is None:
            return None
        if self.problem_type is None:
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"
            self.config.problem_type = self.problem_type
        if self.problem_type == "regression":
            if self.num_labels == 1:
                return F.mse_loss(logits.squeeze(), labels.squeeze())
            logits, labels = logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            return sum(F.mse_loss(logits[:, i], labels[:, i]).sqrt() for i in range(self.num_labels))
        if self.problem_type == "single_label_classification":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        if self.problem_type == "multi_label_classification":
            return F.binary_cross_entropy_with_logits(logits, labels)
        raise ValueError(f"problem_type should be one of {self.problem_types}, but got {self.problem_type}")
