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

import torch
from danling import NestedTensor
from torch import Tensor, nn

from .registry import CRITERIONS

if TYPE_CHECKING:
    from ..heads.config import HeadConfig


@CRITERIONS.register("binary")
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__(**config.get("loss", {}))
        self.config = config

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        if isinstance(input, NestedTensor):
            input = torch.cat(input.flatten().storage())
        if isinstance(target, NestedTensor):
            target = torch.cat(target.flatten().storage())
        if input.ndim == target.ndim + 1:
            input = input.squeeze(-1)
        return super().forward(input, target.float())
