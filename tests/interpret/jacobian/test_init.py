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

from multimolecule.interpret import jacobian


class Quadratic(nn.Module):
    def forward(self, inputs, *, scale):
        first, second = inputs.unbind(dim=-1)
        return {"logits": first + second, "scores": scale * torch.stack((first.square() + second, first * second))}


class TokenPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.lookup = nn.Embedding.from_pretrained(torch.tensor([[1.0, 2.0], [3.0, 4.0], [-2.0, 5.0]]))
        self.calls = 0

    def get_input_embeddings(self):
        return self.lookup

    def forward(self, input_ids, *, mask, scale):
        self.calls += 1
        first, second = self.lookup(input_ids).unbind(dim=-1)
        return {"logits": scale * (input_ids + 1) * first * second * mask}


class DisconnectedPrediction(nn.Module):
    def __init__(self, *, detached):
        super().__init__()
        self.layer = nn.Identity()
        self.weight = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.detached = detached

    def forward(self, inputs):
        value = self.layer(inputs)
        return value.detach() if self.detached else self.weight * inputs


@pytest.fixture
def token_inputs():
    return {"input_ids": torch.tensor([[0, 2, 1]]), "mask": torch.tensor([[True, True, False]])}


class TestJVP:
    def test_projects_selected_output(self):
        output, projected = jacobian.jvp(
            Quadratic(),
            torch.tensor([2.0, 3.0]),
            torch.tensor([3.0, -1.0]),
            output="scores",
            model_kwargs={"scale": 2.0},
        )

        torch.testing.assert_close(output, torch.tensor([14.0, 12.0]))
        torch.testing.assert_close(projected, torch.tensor([22.0, 14.0]))

    def test_rejects_inference_mode(self):
        inputs = torch.tensor([1.0, 2.0])
        with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
            jacobian.jvp(nn.Identity(), inputs, inputs)


class TestVJP:
    def test_projects_onto_each_input(self):
        class Polynomial(nn.Module):
            def forward(self, first, second):
                return {"logits": first.square() + 3 * second}

        output, projected = jacobian.vjp(
            Polynomial(),
            (torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])),
            torch.tensor([2.0, -1.0]),
        )

        torch.testing.assert_close(output, torch.tensor([7.0, 16.0]))
        torch.testing.assert_close(projected, (torch.tensor([4.0, -4.0]), torch.tensor([6.0, -3.0])))

    def test_returns_single_tensor(self):
        output, projected = jacobian.vjp(nn.Identity(), torch.tensor([2.0, 3.0]), torch.tensor([1.0, -2.0]))

        torch.testing.assert_close(output, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(projected, torch.tensor([1.0, -2.0]))

    @pytest.mark.parametrize("layer", ["embedding", "lookup", "module"], ids=["embedding", "path", "module"])
    def test_projects_onto_frozen_embeddings(self, layer, token_inputs):
        model = TokenPrediction()
        with torch.no_grad():
            output, projected = jacobian.vjp(
                model,
                token_inputs,
                torch.tensor([[2.0, -1.0, 7.0]]),
                wrt=model.lookup if layer == "module" else layer,
                model_kwargs={"scale": 2.0},
            )

        torch.testing.assert_close(output, torch.tensor([[4.0, -60.0, 0.0]]))
        torch.testing.assert_close(projected, torch.tensor([[[8.0, 4.0], [-30.0, 12.0], [0.0, 0.0]]]))

    @pytest.mark.parametrize("detached", [True, False], ids=["detached", "independent"])
    def test_zeros_disconnected_derivatives(self, detached):
        model = DisconnectedPrediction(detached=detached)
        inputs = torch.tensor([1.0, 2.0])

        _, projected = jacobian.vjp(model, inputs, torch.ones(2), wrt=model.layer)

        torch.testing.assert_close(projected, torch.zeros(2))

    def test_excludes_paths_bypassing_layer(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Identity()

            def forward(self, inputs):
                return 2 * self.layer(inputs) + 3 * inputs

        model = Residual()
        inputs = torch.tensor([1.0, 2.0], requires_grad=True)
        output, projected = jacobian.vjp(model, inputs, torch.ones(2), wrt=model.layer)

        torch.testing.assert_close(output, torch.tensor([5.0, 10.0]))
        torch.testing.assert_close(projected, torch.tensor([2.0, 2.0]))

    def test_differentiates_through_frozen_backbone(self):
        class FrozenBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Identity()

            def forward(self, inputs):
                with torch.no_grad():
                    activation = self.layer(inputs)
                return activation.square()

        model = FrozenBackbone()
        inputs = torch.tensor([1.0, 2.0])
        output, projected = jacobian.vjp(model, inputs, torch.ones(2), wrt=model.layer)

        torch.testing.assert_close(output, torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(projected, torch.tensor([2.0, 4.0]))


class TestFull:
    @pytest.fixture
    def repeated_layer(self):
        class Repeated(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Identity()

            def forward(self, inputs):
                return self.layer(inputs) + self.layer(inputs)

        return Repeated()

    def test_orders_output_axes_first(self):
        result = jacobian.full(
            Quadratic(),
            torch.tensor([2.0, 3.0]),
            output=lambda value: value["scores"],
            model_kwargs={"scale": 2.0},
            chunk_size=1,
        )

        torch.testing.assert_close(result, torch.tensor([[8.0, 2.0], [6.0, 4.0]]))

    @pytest.mark.parametrize("chunk_size", [1, 2], ids=["single-row", "two-rows"])
    def test_differentiates_selected_output_rows(self, chunk_size, token_inputs):
        model = TokenPrediction()

        with torch.no_grad():
            result = jacobian.full(
                model,
                token_inputs,
                wrt="embedding",
                output=lambda value: value["logits"][..., :2],
                model_kwargs={"scale": 2.0},
                chunk_size=chunk_size,
            )

        expected = torch.tensor(
            [[[4.0, 2.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [30.0, -12.0], [0.0, 0.0]]]
        ).reshape(1, 2, 1, 3, 2)
        torch.testing.assert_close(result, expected)

    def test_runs_model_once(self, token_inputs):
        model = TokenPrediction()
        jacobian.full(model, token_inputs, wrt="embedding", model_kwargs={"scale": 2.0}, chunk_size=1)

        assert model.calls == 1

    @pytest.mark.parametrize("detached", [True, False], ids=["detached", "independent"])
    def test_zeros_disconnected_derivatives(self, detached):
        model = DisconnectedPrediction(detached=detached)
        result = jacobian.full(model, torch.tensor([1.0, 2.0]), wrt=model.layer, chunk_size=2)

        torch.testing.assert_close(result, torch.zeros(2, 2))

    def test_requires_real_output(self):
        class ComplexPrediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Identity()

            def forward(self, inputs):
                activation = self.layer(inputs)
                return torch.complex(activation, 2 * activation)

        model = ComplexPrediction()
        with pytest.raises(TypeError, match="output.*floating-point"):
            jacobian.full(model, torch.tensor([1.0, 2.0]), wrt=model.layer)

    def test_rejects_reused_layer(self, repeated_layer):
        with pytest.raises(RuntimeError, match="more than once"):
            jacobian.full(repeated_layer, torch.tensor([1.0, 2.0]), wrt=repeated_layer.layer)

    def test_restores_model_after_error(self, repeated_layer):
        inputs = torch.tensor([1.0, 2.0])
        with pytest.raises(RuntimeError, match="more than once"):
            jacobian.full(repeated_layer, inputs, wrt=repeated_layer.layer)

        output = repeated_layer(inputs)
        torch.testing.assert_close(output, torch.tensor([2.0, 4.0]))
        assert not output.requires_grad
