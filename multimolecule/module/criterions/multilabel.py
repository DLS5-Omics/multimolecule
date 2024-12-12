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

from typing import TYPE_CHECKING

import torch
from danling import NestedTensor
from torch import Tensor, nn

if TYPE_CHECKING:
    from ..heads.config import HeadConfig

from .registry import CriterionRegistry


@CriterionRegistry.register("multilabel")
class MultiLabelSoftMarginLoss(nn.MultiLabelSoftMarginLoss):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__(**config.get("loss", {}))
        self.config = config

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        if isinstance(target, NestedTensor) and target.ndim > 2:
            input, target = input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
        if isinstance(input, NestedTensor):
            input = torch.cat(input.storage())
        if isinstance(target, NestedTensor):
            target = torch.cat(target.storage())
        return super().forward(input, target.float())
