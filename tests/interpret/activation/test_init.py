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


class TestCapture:
    def test_returns_ordered_snapshots(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Identity()

            def forward(self, *, values, scale):
                first = self.layer(values * scale)
                first.add_(100)
                return self.layer(first + 1)

        values = torch.tensor([1.0, 2.0])
        model = Model()
        result = capture(model, {"values": values}, layers=[model.layer, "layer"], model_kwargs={"scale": 2})

        assert tuple(result) == ("layer",)
        torch.testing.assert_close(result["layer"], (torch.tensor([2.0, 4.0]), torch.tensor([103.0, 105.0])))

    def test_detaches_by_default(self):
        model = nn.Sequential(nn.Identity())
        values = torch.tensor([1.0, 2.0], requires_grad=True)

        (snapshot,) = capture(model, values, layers="0")["0"]

        assert not snapshot.requires_grad

    def test_preserves_structured_gradients(self):
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

    def test_preserves_existing_observers(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Identity()

            def forward(self, values, *, fail=False):
                result = self.layer(values)
                if fail:
                    raise RuntimeError("forward failed")
                return result

        model = Model()
        observed = []
        values = torch.tensor([1.0])

        def observe(_module, _args, output):
            observed.append(output)

        with model.layer.register_forward_hook(observe):
            with pytest.raises(RuntimeError, match="forward failed"):
                capture(model, values, layers="layer", model_kwargs={"fail": True})
            model(values + 1)

        torch.testing.assert_close(observed, [values, values + 1])
