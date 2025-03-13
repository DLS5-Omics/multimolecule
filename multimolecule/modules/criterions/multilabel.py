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


@CRITERIONS.register("multilabel")
class MultiLabelSoftMarginLoss(nn.MultiLabelSoftMarginLoss):
    r"""
    Multi-label classification loss that supports NestedTensor and ignore_index.

    Attributes:
        ignore_index: Value to ignore in the target tensor. If None, no values are ignored.
            Defaults to -100.

    Examples:
        >>> import torch
        >>> from ..heads.config import HeadConfig
        >>> criterion = MultiLabelSoftMarginLoss(HeadConfig())
        >>> input = torch.tensor([[0.6, -0.5], [0.7, 0.3]])
        >>> target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.5423)
        >>> assert loss == torch.nn.functional.multilabel_soft_margin_loss(input, target)
        >>> input = torch.tensor([[0.6, -0.5], [0.7, 0.3]])
        >>> target = torch.tensor([[1.0, 0.0], [1.0, -100]])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.4558)
    """

    ignore_index: int | None = None

    def __init__(self, config: HeadConfig) -> None:
        loss_config = config.get("loss", {})
        ignore_index = loss_config.pop("ignore_index", -100)
        super().__init__(**loss_config)
        self.config = config
        self.ignore_index = ignore_index

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        if isinstance(target, NestedTensor) and target.ndim > 2:
            input, target = input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
        if isinstance(input, NestedTensor):
            input = torch.cat(input.storage())
        if isinstance(target, NestedTensor):
            target = torch.cat(target.storage())
        if self.ignore_index is not None:
            # For multilabel, we need to check across all labels
            # Create a mask for samples where no label is set to ignore_index
            mask = (target != self.ignore_index).all(dim=-1)
            input = input[mask]
            target = target[mask]
        return super().forward(input, target.float())
