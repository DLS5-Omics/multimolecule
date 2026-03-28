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

import torch
from danling import NestedTensor

from multimolecule.modules.layers import SymmetricConv2d, SymmetricConvTranspose2d
from multimolecule.modules.networks import ResNet


class TestSymmetricConv2d:
    def test_forward(self):
        layer = SymmetricConv2d(4, 2, kernel_size=3, padding=1)
        x = torch.randn(2, 4, 5, 5)

        output = layer(x)

        assert output.shape == (2, 2, 5, 5)
        assert torch.allclose(output, output.transpose(-1, -2))

    def test_forward_with_nested_tensor(self):
        layer = SymmetricConv2d(4, 2, kernel_size=3, padding=1)
        x = NestedTensor([torch.randn(4, 5, 5), torch.randn(4, 3, 3)])

        output = layer(x)

        assert isinstance(output, NestedTensor)
        assert output.tensor.shape == (2, 2, 5, 5)
        for item in output:
            assert torch.allclose(item, item.transpose(-1, -2))


class TestSymmetricConvTranspose2d:
    def test_forward(self):
        layer = SymmetricConvTranspose2d(4, 2, kernel_size=3, padding=1)
        x = torch.randn(2, 4, 5, 5)

        output = layer(x)

        assert output.shape == (2, 2, 5, 5)
        assert torch.allclose(output, output.transpose(-1, -2))

    def test_forward_with_nested_tensor(self):
        layer = SymmetricConvTranspose2d(4, 2, kernel_size=3, padding=1)
        x = NestedTensor([torch.randn(4, 5, 5), torch.randn(4, 3, 3)])

        output = layer(x)

        assert isinstance(output, NestedTensor)
        assert output.tensor.shape == (2, 2, 5, 5)
        for item in output:
            assert torch.allclose(item, item.transpose(-1, -2))


def test_symmetric_resnet_forward_with_nested_tensor():
    model = ResNet(num_layers=2, hidden_size=8, num_channels=2, symmetric=True)
    x = NestedTensor([torch.randn(5, 5, 8), torch.randn(3, 3, 8)])

    output = model(x)

    assert isinstance(output, NestedTensor)
    for item in output:
        assert torch.allclose(item, item.transpose(-2, -3))
