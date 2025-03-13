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
    r"""
    Mean Squared Error Loss that supports NestedTensor and ignoring NaN values.

    Attributes:
        ignore_nan: Whether to ignore NaN values in the target tensor.
            Defaults to True.

    Examples:
        >>> import torch
        >>> from ..heads.config import HeadConfig
        >>> criterion = MSELoss(HeadConfig())
        >>> input = torch.tensor([0.5, 1.2, 3.4, 2.0])
        >>> target = torch.tensor([0.7, 1.0, 3.5, 2.5])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.0850)
        >>> assert loss == torch.nn.functional.mse_loss(input, target)
        >>> input = torch.tensor([0.5, 1.2, 3.4, 2.0])
        >>> target = torch.tensor([0.7, 1.0, float('nan'), float('nan')])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.0400)
        >>> assert loss == torch.nn.functional.mse_loss(input[:2], target[:2])
    """

    ignore_nan: bool = True

    def __init__(self, config: HeadConfig) -> None:
        loss_config = config.get("loss", {})
        self.ignore_nan = loss_config.pop("ignore_nan", True)
        super().__init__(**loss_config)
        self.config = config

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        if isinstance(input, NestedTensor):
            input = torch.cat(input.flatten().storage())
        if isinstance(target, NestedTensor):
            target = torch.cat(target.flatten().storage())
        if input.ndim == target.ndim + 1:
            target = target.unsqueeze(-1)
        if self.ignore_nan:
            mask = ~torch.isnan(target)
            input = input[mask]
            target = target[mask]
        return super().forward(input, target.to(input.dtype))


@CRITERIONS.register("rmse")
class RMSELoss(MSELoss):
    r"""
    Root Mean Squared Error Loss that supports NestedTensor and ignoring NaN values.

    Attributes:
        ignore_nan: Whether to ignore NaN values in the target tensor.
            Defaults to True.

    Examples:
        >>> import torch
        >>> from ..heads.config import HeadConfig
        >>> criterion = RMSELoss(HeadConfig())
        >>> input = torch.tensor([0.5, 1.2, 3.4, 2.0])
        >>> target = torch.tensor([0.7, 1.0, 3.5, 2.5])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.2915)
        >>> assert loss == torch.nn.functional.mse_loss(input, target).sqrt()
        >>> input = torch.tensor([0.5, 1.2, 3.4, 2.0])
        >>> target = torch.tensor([0.7, 1.0, float('nan'), float('nan')])
        >>> loss = criterion(input, target)
        >>> loss
        tensor(0.2000)
        >>> assert loss == torch.nn.functional.mse_loss(input[:2], target[:2]).sqrt()
    """

    def forward(self, input: NestedTensor | Tensor, target: NestedTensor | Tensor) -> Tensor:
        mse = super().forward(input, target)
        return torch.sqrt(mse)
