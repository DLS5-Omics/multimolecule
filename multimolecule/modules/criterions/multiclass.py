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


@CRITERIONS.register("multiclass")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    r"""
    Cross Entropy Loss that supports NestedTensor inputs.

    Examples:
        >>> import torch
        >>> from ..heads.config import HeadConfig
        >>> criterion = CrossEntropyLoss(HeadConfig())
        >>> input = torch.tensor([[0.1, 0.2, 0.9], [1.1, 0.1, 0.2]])
        >>> target = torch.tensor([2, 0])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.6196)
        >>> assert loss == torch.nn.functional.cross_entropy(input, target)
        >>> input = torch.tensor([[0.1, 0.2, 0.9], [1.1, 0.1, 0.2]])
        >>> target = torch.tensor([2, -100])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.6657)
        >>> assert loss == torch.nn.functional.cross_entropy(input[:1], target[:1])
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__(**config.get("loss", {}))
        self.config = config

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        if isinstance(input, NestedTensor):
            input = torch.cat(input.storage())
        if isinstance(target, NestedTensor):
            target = torch.cat(target.storage())
        if input.ndim > 2:
            input, target = input.view(-1, input.size(-1)), target.view(-1)
        return super().forward(input, target.long())
