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

if TYPE_CHECKING:
    from ..heads.config import HeadConfig

from .registry import CRITERIONS


@CRITERIONS.register("regression")
class MSELoss(nn.MSELoss):
    ignore_nan: bool = True

    def __init__(self, config: HeadConfig) -> None:
        loss_config = dict(config.get("loss", {}))
        self.ignore_nan = loss_config.pop("ignore_nan", True)
        super().__init__(**loss_config)
        self.config = config
        self.num_labels = config.num_labels

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        if input.ndim == target.ndim + 1:
            input = input.squeeze(-1)
        if self.ignore_nan:
            mask = ~torch.isnan(target)
            if not bool(mask.all()):
                if isinstance(target, NestedTensor) and not isinstance(input, NestedTensor):
                    input = target.nested_like(input, strict=False)
                input = input[mask]
                target = target[mask]
        return super().forward(input, target.to(input.dtype))


@CRITERIONS.register("rmse")
class RMSELoss(MSELoss):
    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        return torch.sqrt(super().forward(input, target))
