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

from multimolecule.interpret import attribution

pytestmark = pytest.mark.usefixtures("captum")


class ScoredPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[2.0, -3.0]]))

    def forward(self, *, values, multiplier):
        hidden = self.hidden(values * multiplier)
        return {"scores": torch.cat((hidden, 5 * hidden), dim=-1)}


class TestNeuronIntegratedGradients:
    @pytest.mark.parametrize(
        ("selector", "to_input", "expected"),
        [
            (1, False, [[-2.0, -3.0], [3.0, 6.0]]),
            ((slice(None),), False, [[2.0, -4.0], [-3.0, 8.0]]),
            (1, True, [[0.0, -1.0], [0.0, 2.0]]),
        ],
        ids=["output-index", "output-slice", "input-index"],
    )
    def test_explains_selected_neuron(self, selector, to_input, expected):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.hidden.weight.copy_(torch.tensor([[2.0, 1.0], [-1.0, 3.0]]))

            def forward(self, values):
                self.hidden(values)
                return {"metadata": "opaque"}

        model = Prediction()
        result = attribution.neuron_integrated_gradients(
            model,
            torch.tensor([[3.0, 0.0], [-2.0, 3.0]]),
            layer="hidden",
            neuron_selector=selector,
            baselines=1,
            n_steps=4,
            attribute_to_neuron_input=to_input,
        )

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert result.convergence_delta is None

    def test_explains_named_inputs(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.hidden.weight.copy_(torch.tensor([[2.0, 1.0]]))

            def forward(self, *, values, multiplier):
                self.hidden(values * multiplier)
                return {"metadata": "opaque"}

        model = Prediction()
        result = attribution.neuron_integrated_gradients(
            model,
            {"values": torch.tensor([[3.0, 0.0], [-2.0, 3.0]]), "multiplier": torch.tensor([[2.0], [0.5]])},
            layer=model.hidden,
            neuron_selector=lambda hidden: hidden[:, 0],
            input_keys="values",
            baselines=1,
            n_steps=4,
            internal_batch_size=2,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[8.0, -2.0], [-3.0, 1.0]]))


class TestNeuronScoreAttribution:
    @pytest.mark.parametrize(
        ("attribute", "options", "expected"),
        [
            (attribution.neuron_gradient, {}, [[4.0, -6.0], [1.0, -1.5]]),
            (
                attribution.neuron_conductance,
                {"output": "scores", "target": 1, "method": "gausslegendre"},
                [[60.0, -120.0], [-10.0, -45.0]],
            ),
            (attribution.neuron_deep_lift, {"output": "scores", "baselines": 0}, [[12.0, -24.0], [-2.0, -9.0]]),
            (
                attribution.neuron_deep_lift_shap,
                {"output": "scores", "baselines": torch.tensor([[0.0, 0.0], [2.0, 2.0]])},
                [[8.0, -18.0], [-3.0, -7.5]],
            ),
            (
                attribution.neuron_gradient_shap,
                {"baselines": torch.zeros(2, 2), "n_samples": 2},
                [[12.0, -24.0], [-2.0, -9.0]],
            ),
        ],
        ids=["gradient", "conductance", "deep-lift", "deep-lift-shap", "gradient-shap"],
    )
    def test_explains_named_inputs(self, attribute, options, expected):
        model = ScoredPrediction()
        values = torch.tensor([[3.0, 4.0], [-2.0, 6.0]])
        inputs = {"values": values, "multiplier": torch.tensor([[2.0], [0.5]])}

        result = attribute(model, inputs, layer=model.hidden, neuron_selector=0, input_keys="values", **options)

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert result.convergence_delta is None
        assert not values.requires_grad


class TestNeuronFeatureAblation:
    def test_ablates_grouped_tokens(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Identity()
                self.register_buffer("weights", torch.arange(6, dtype=torch.float32))

            def forward(self, *, ids, multiplier):
                self.hidden(self.weights[ids].sum(dim=-1, keepdim=True) * multiplier)
                return {"metadata": "opaque"}

        model = Prediction()
        result = attribution.neuron_feature_ablation(
            model,
            {"ids": torch.tensor([[1, 2, 3], [3, 4, 5]]), "multiplier": torch.tensor([[2.0], [1.0]])},
            input_keys="ids",
            layer="hidden",
            neuron_selector=0,
            baselines=0,
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=2,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[6.0, 6.0, 6.0], [7.0, 7.0, 5.0]]))


class TestNeuronReluAttribution:
    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [
            (attribution.neuron_deconvolution, [[3.0, 0.0], [3.0, 0.0]]),
            (attribution.neuron_guided_backprop, [[0.0, 0.0], [3.0, 0.0]]),
        ],
        ids=["deconvolution", "guided-backprop"],
    )
    def test_filters_inactive_inputs(self, attribute, expected):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = nn.ReLU()
                self.hidden = nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.hidden.weight.copy_(torch.tensor([[3.0, -2.0]]))

            def forward(self, *, values):
                self.hidden(self.activation(values))
                return {"metadata": "opaque"}

        model = Prediction()
        values = torch.tensor([[-2.0, 3.0], [1.0, -4.0]])
        result = attribute(model, {"values": values}, input_keys="values", layer="hidden", neuron_selector=0)

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert not values.requires_grad

    @pytest.mark.parametrize(
        "attribute",
        [attribution.neuron_guided_backprop, attribution.neuron_deconvolution],
        ids=["guided-backprop", "deconvolution"],
    )
    def test_preserves_input_gradients(self, attribute, weighted_relu):
        inputs = torch.tensor([[1.0, -1.0, 1.0]], requires_grad=True)

        attribute(weighted_relu, inputs, layer=weighted_relu.readout, neuron_selector=0)

        (gradient,) = torch.autograd.grad(weighted_relu(inputs).sum(), inputs)
        torch.testing.assert_close(gradient, torch.tensor([[-2.0, 0.0, 4.0]]))


class TestNeuronDeepLift:
    def test_recovers_from_output_error(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = nn.ReLU()
                self.hidden = nn.Identity()

            def forward(self, values):
                return {"scores": self.hidden(self.activation(values))}

        model = Prediction()
        original_output = object()
        model.activation.output = original_output
        values = torch.tensor([[-2.0, 3.0], [1.0, -4.0]])

        with pytest.raises(TypeError, match="select a tensor"):
            attribution.neuron_deep_lift(model, values, layer="hidden", neuron_selector=0)

        assert model.activation.output is original_output
        assert not values.requires_grad

        result = attribution.neuron_deep_lift(model, values, layer="hidden", neuron_selector=0, output="scores")

        torch.testing.assert_close(result.attributions, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
