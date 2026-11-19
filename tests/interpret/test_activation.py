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

from multimolecule.interpret.activation import capture


def test_capture_keeps_each_layer_invocation_before_in_place_changes():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, *, values, scale):
            first = self.layer(values * scale)
            first.add_(100)
            return self.layer(first + 1)

    values = torch.tensor([1.0, 2.0], requires_grad=True)
    model = Model()
    result = capture(model, {"values": values}, layers=[model.layer, "layer"], model_kwargs={"scale": 2})

    assert tuple(result) == ("layer",)
    torch.testing.assert_close(result["layer"], (torch.tensor([2.0, 4.0]), torch.tensor([103.0, 105.0])))
    assert not any(output.requires_grad for output in result["layer"])


def test_capture_preserves_gradients_in_structured_layer_output():
    class Layer(nn.Module):
        def forward(self, values):
            return ({"scaled": values * 3}, [values + 1])

    values = torch.tensor([1.0, 2.0], requires_grad=True)
    model = nn.Sequential(nn.Sequential(Layer()))
    result = capture(model, values, layers=model[0][0], detach=False)

    mapping, sequence = result["0.0"][0]
    torch.testing.assert_close(mapping["scaled"], torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(sequence[0], torch.tensor([2.0, 3.0]))
    mapping["scaled"].sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([3.0, 3.0]))


def test_capture_removes_its_hook_after_model_error():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, values):
            self.layer(values)
            raise RuntimeError("forward failed")

    model = Model()
    with model.layer.register_forward_hook(lambda *_: None) as existing_hook:
        with pytest.raises(RuntimeError, match="forward failed"):
            capture(model, torch.tensor([1.0]), layers="layer")
        assert set(model.layer._forward_hooks) == {existing_hook.id}
