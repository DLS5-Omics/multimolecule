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

import pytest
import torch
from torch import nn

from multimolecule.interpret.attention import capture


def test_capture_attention_selects_layers_and_heads_with_custom_head_axis():
    class Model(nn.Module):
        def forward(self, values, *, output_attentions=False):
            return {"attentions": (values, values + 100) if output_attentions else None}

    values = torch.arange(48.0).reshape(2, 2, 3, 2, 2)
    maps = capture(
        Model(),
        values,
        layers=[-1, 0],
        heads=[2, 0],
        head_dim=2,
        model_kwargs={"output_attentions": True},
    )

    torch.testing.assert_close(maps, (values[:, :, [2, 0]] + 100, values[:, :, [2, 0]]))


def test_capture_attention_selects_tensor_without_squeezing_head_axis():
    class Model(nn.Module):
        def forward(self, values):
            return {"weights": values}

    values = torch.arange(12.0).reshape(1, 3, 2, 2)
    maps = capture(Model(), values, heads=-1, output=lambda result: result["weights"])

    torch.testing.assert_close(maps, (values[:, -1:],))


def test_capture_attention_explains_how_to_request_missing_attentions():
    with pytest.raises(ValueError, match="output_attentions"):
        capture(nn.Identity(), torch.zeros(1, 2, 3))
