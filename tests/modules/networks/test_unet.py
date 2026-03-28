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

import pytest
import torch

from multimolecule.modules.networks import UNet


class TestUNet:
    def test_forward_with_small_channel_count(self):
        model = UNet(num_layers=2, hidden_size=8, num_channels=2)
        x = torch.randn(2, 6, 6, 8)

        output = model(x)

        assert output.shape == (2, 6, 6, 1)

    def test_symmetric_forward(self):
        model = UNet(num_layers=2, hidden_size=8, num_channels=2, symmetric=True)
        x = torch.randn(2, 6, 6, 8)

        output = model(x)

        assert torch.allclose(output, output.transpose(1, 2))

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"num_layers": 0, "hidden_size": 8}, "num_layers"),
            ({"num_layers": 3, "hidden_size": 8}, "even number"),
            ({"num_layers": 2, "hidden_size": 0}, "hidden_size"),
            ({"num_layers": 2, "hidden_size": 8, "num_channels": 0}, "num_channels"),
        ],
    )
    def test_invalid_configuration_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            UNet(**kwargs)
