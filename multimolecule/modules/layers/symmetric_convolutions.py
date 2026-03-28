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

from torch import Tensor, nn


class SymmetricConv2d(nn.Conv2d):

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for a convolution layer whose output is symmetrized over the last two dimensions.

        Args:
            input: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            nn.Tensor: Symmetric output tensor after convolution.
        """
        output = super().forward(input)
        return (output + output.transpose(-1, -2)) / 2


class SymmetricConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for a transposed convolution layer whose output is symmetrized over the last two dimensions.

        Args:
            input: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            nn.Tensor: Symmetric output tensor after transposed convolution.
        """
        output = super().forward(input)
        return (output + output.transpose(-1, -2)) / 2
