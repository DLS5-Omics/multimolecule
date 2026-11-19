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

from multimolecule.interpret import jacobian as derivatives


class Quadratic(nn.Module):
    def forward(self, inputs, *, scale):
        first, second = inputs.unbind(dim=-1)
        return {"logits": first + second, "scores": scale * torch.stack((first.square() + second, first * second))}


def test_jvp_projects_the_selected_model_output():
    output, projected = derivatives.jvp(
        Quadratic(),
        torch.tensor([2.0, 3.0]),
        torch.tensor([3.0, -1.0]),
        output="scores",
        model_kwargs={"scale": 2.0},
    )

    torch.testing.assert_close(output, torch.tensor([14.0, 12.0]))
    torch.testing.assert_close(projected, torch.tensor([22.0, 14.0]))


def test_vjp_returns_one_cotangent_per_input():
    class Polynomial(nn.Module):
        def forward(self, first, second):
            return {"logits": first.square() + 3 * second}

    output, projected = derivatives.vjp(
        Polynomial(),
        (torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])),
        torch.tensor([2.0, -1.0]),
    )

    torch.testing.assert_close(output, torch.tensor([7.0, 16.0]))
    torch.testing.assert_close(projected, (torch.tensor([4.0, -4.0]), torch.tensor([6.0, -3.0])))


def test_vjp_unwraps_the_cotangent_for_a_single_tensor():
    output, projected = derivatives.vjp(nn.Identity(), torch.tensor([2.0, 3.0]), torch.tensor([1.0, -2.0]))

    torch.testing.assert_close(output, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(projected, torch.tensor([1.0, -2.0]))


def test_chunked_jacobian_has_output_axes_before_input_axes():
    result = derivatives.full(
        Quadratic(),
        torch.tensor([2.0, 3.0]),
        output=lambda value: value["scores"],
        model_kwargs={"scale": 2.0},
        chunk_size=1,
    )

    torch.testing.assert_close(result, torch.tensor([[8.0, 2.0], [6.0, 4.0]]))


def test_differentiation_rejects_inference_mode():
    inputs = torch.tensor([1.0, 2.0])
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        derivatives.jvp(nn.Identity(), inputs, inputs)


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


@pytest.mark.parametrize("layer", ["embedding", "lookup", "module"])
def test_layer_vjp_uses_original_integer_inputs_and_frozen_embedding(layer):
    model = TokenPrediction()
    inputs = {"input_ids": torch.tensor([[0, 2, 1]]), "mask": torch.tensor([[True, True, False]])}
    with torch.no_grad():
        output, projected = derivatives.vjp(
            model,
            inputs,
            torch.tensor([[2.0, -1.0, 7.0]]),
            wrt=model.lookup if layer == "module" else layer,
            model_kwargs={"scale": 2.0},
        )

    torch.testing.assert_close(output, torch.tensor([[4.0, -60.0, 0.0]]))
    torch.testing.assert_close(projected, torch.tensor([[[8.0, 4.0], [-30.0, 12.0], [0.0, 0.0]]]))
    assert model.calls == 1


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_layer_jacobian_uses_one_forward_for_selected_output_rows(chunk_size):
    model = TokenPrediction()
    inputs = {"input_ids": torch.tensor([[0, 2, 1]]), "mask": torch.tensor([[True, True, False]])}

    with torch.no_grad():
        result = derivatives.full(
            model,
            inputs,
            wrt="embedding",
            output=lambda value: value["logits"][..., :2],
            model_kwargs={"scale": 2.0},
            chunk_size=chunk_size,
        )

    assert result.shape == (1, 2, 1, 3, 2)
    expected = torch.tensor([[[4.0, 2.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [30.0, -12.0], [0.0, 0.0]]])
    torch.testing.assert_close(result.reshape(2, 3, 2), expected)
    assert model.calls == 1


@pytest.mark.parametrize("detached", [True, False], ids=["detached", "independent"])
def test_layer_derivatives_are_zero_for_outputs_without_a_gradient_path(detached):
    class NoGradientPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()
            self.weight = nn.Parameter(torch.tensor([3.0, 4.0]))

        def forward(self, inputs):
            value = self.layer(inputs)
            return value.detach() if detached else self.weight * inputs

    model = NoGradientPath()
    inputs = torch.tensor([1.0, 2.0])
    _, projected = derivatives.vjp(model, inputs, torch.ones(2), wrt=model.layer)
    result = derivatives.full(model, inputs, wrt=model.layer, chunk_size=2)

    torch.testing.assert_close(projected, torch.zeros(2))
    torch.testing.assert_close(result, torch.zeros(2, 2))


def test_layer_vjp_excludes_paths_that_bypass_an_identity_layer():
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, inputs):
            return 2 * self.layer(inputs) + 3 * inputs

    model = Residual()
    inputs = torch.tensor([1.0, 2.0], requires_grad=True)
    output, projected = derivatives.vjp(model, inputs, torch.ones(2), wrt=model.layer)

    torch.testing.assert_close(output, torch.tensor([5.0, 10.0]))
    torch.testing.assert_close(projected, torch.tensor([2.0, 2.0]))


def test_layer_vjp_connects_a_frozen_backbone_activation_to_the_head():
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
    output, projected = derivatives.vjp(model, inputs, torch.ones(2), wrt=model.layer)

    torch.testing.assert_close(output, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(projected, torch.tensor([2.0, 4.0]))


def test_layer_jacobian_requires_a_floating_point_output():
    class ComplexPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, inputs):
            activation = self.layer(inputs)
            return torch.complex(activation, 2 * activation)

    model = ComplexPrediction()
    with pytest.raises(TypeError, match="output.*floating-point"):
        derivatives.full(model, torch.tensor([1.0, 2.0]), wrt=model.layer)


def test_layer_derivatives_reject_repeated_calls_and_cleanup_hooks():
    class Repeated(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, inputs):
            return self.layer(inputs) + self.layer(inputs)

    model = Repeated()
    inputs = torch.tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="more than once"):
        derivatives.full(model, inputs, wrt=model.layer)

    torch.testing.assert_close(model(inputs), torch.tensor([2.0, 4.0]))
