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

from multimolecule.interpret.attribution import (
    deconvolution,
    deep_lift,
    deep_lift_shap,
    guided_backprop,
    guided_grad_cam,
    lrp,
)

pytestmark = pytest.mark.usefixtures("captum")


class ReLUScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, *, values, offset):
        return {"scores": self.relu(values).sum(dim=-1) + offset}


class TestDeepLift:
    def test_explains_named_inputs(self):
        result = deep_lift(
            ReLUScore(),
            {"values": torch.tensor([[2.0, -3.0]]), "offset": torch.tensor([5.0])},
            input_keys="values",
            baselines=torch.tensor([[-2.0, -1.0]]),
            output="scores",
            return_convergence_delta=True,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[2.0, 0.0]]))
        torch.testing.assert_close(result.convergence_delta, torch.zeros(1))

    def test_preserves_caller_attribute(self):
        relu = nn.ReLU()
        original_input = object()
        relu.input = original_input

        deep_lift(nn.Sequential(relu), torch.tensor([[2.0, -1.0]]), target=0)

        assert relu.input is original_input

    def test_preserves_nonpersistent_buffer(self):
        relu = nn.ReLU()
        buffer = torch.tensor([7.0])
        relu.register_buffer("input", buffer, persistent=False)

        deep_lift(nn.Sequential(relu), torch.tensor([[2.0, -1.0]]), target=0)

        assert relu.get_buffer("input") is buffer
        assert "input" not in relu.state_dict()


class TestDeepLiftShap:
    def test_averages_baselines(self):
        result = deep_lift_shap(
            ReLUScore(),
            {"values": torch.tensor([[2.0, -3.0]]), "offset": torch.tensor([5.0])},
            input_keys="values",
            baselines=torch.tensor([[-2.0, -1.0], [1.0, -2.0]]),
            output="scores",
            return_convergence_delta=True,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[1.5, 0.0]]))
        torch.testing.assert_close(result.convergence_delta, torch.zeros(2))


class TestPairedBaselines:
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
    @pytest.mark.parametrize("attribute", [deep_lift, deep_lift_shap], ids=["deep-lift", "deep-lift-shap"])
    def test_rejects_multiple_replicas(self, attribute):
        model = nn.DataParallel(nn.ReLU().cuda(), device_ids=[0, 1])
        inputs = torch.ones(2, 2, device="cuda")

        with pytest.raises(ValueError, match="single model replica"):
            attribute(model, inputs, baselines=torch.zeros_like(inputs))


class TestReluAttribution:
    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [(guided_backprop, [[0.0, 0.0, 4.0]]), (deconvolution, [[0.0, 3.0, 4.0]])],
        ids=["guided-backprop", "deconvolution"],
    )
    def test_filters_signed_gradients(self, attribute, expected, weighted_relu):
        result = attribute(weighted_relu, torch.tensor([[1.0, -1.0, 1.0]]))

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert result.convergence_delta is None

    @pytest.mark.parametrize("attribute", [guided_backprop, deconvolution], ids=["guided-backprop", "deconvolution"])
    def test_preserves_input_gradients(self, attribute, weighted_relu):
        inputs = torch.tensor([[1.0, -1.0, 1.0]], requires_grad=True)

        attribute(weighted_relu, inputs)

        (gradient,) = torch.autograd.grad(weighted_relu(inputs).sum(), inputs)
        torch.testing.assert_close(gradient, torch.tensor([[-2.0, 0.0, 4.0]]))


class TestGuidedGradCam:
    def test_localizes_guided_gradient(self):
        class ImageScore(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1, bias=False)
                self.relu = nn.ReLU()
                nn.init.constant_(self.conv.weight, 2.0)

            def forward(self, inputs):
                return self.relu(self.conv(inputs)).sum(dim=(1, 2, 3))

        inputs = torch.tensor([[[[1.0, -1.0], [2.0, 3.0]]]])
        result = guided_grad_cam(ImageScore(), inputs, layer="conv")

        torch.testing.assert_close(result.attributions, torch.tensor([[[[3.0, 0.0], [6.0, 9.0]]]]))
        assert result.convergence_delta is None


class TestLrp:
    @pytest.fixture
    def named_score(self, epsilon_rule):
        class Score(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1, bias=False)

            def forward(self, *, values, mode):
                if mode != "regression":
                    raise ValueError("Unknown score type")
                return {"scores": self.linear(values)}

        model = Score()
        model.linear.rule = epsilon_rule()
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[2.0, 3.0]]))
        batch = {"values": torch.tensor([[1.0, 2.0]]), "mode": "regression"}
        return model, batch

    def test_propagates_selected_score(self, named_score):
        model, batch = named_score
        result = lrp(model, batch, input_keys="values", output="scores", return_convergence_delta=True)

        torch.testing.assert_close(result.attributions, torch.tensor([[2.0, 6.0]]))
        torch.testing.assert_close(result.convergence_delta, torch.zeros(1))

    def test_preserves_model_state(self, named_score):
        model, batch = named_score
        rule = model.linear.rule
        weights = model.linear.weight.detach().clone()

        lrp(model, batch, input_keys="values", output="scores")

        torch.testing.assert_close(model.linear.weight, weights)
        assert model.linear.rule is rule

    def test_recovers_from_unsupported_layers(self):
        linear = nn.Linear(2, 1, bias=False)
        nn.init.ones_(linear.weight)
        model = nn.Sequential(linear, nn.GELU())
        inputs = torch.tensor([[1.0, 2.0]])

        with pytest.raises(TypeError, match="no rule defined"):
            lrp(model, inputs)

        result = lrp(nn.Sequential(linear), inputs)

        torch.testing.assert_close(result.attributions, inputs)

    def test_preserves_registered_buffer(self):
        linear = nn.Linear(2, 1)
        buffer = torch.tensor([7.0])
        linear.register_buffer("activation", buffer)

        lrp(nn.Sequential(linear), torch.tensor([[1.0, 2.0]]))

        assert linear.get_buffer("activation") is buffer
        torch.testing.assert_close(linear.state_dict()["activation"], buffer)

    def test_restores_weights(self, mutating_lrp_model):
        inputs = torch.tensor([[1.0, 2.0]])
        original = mutating_lrp_model[0].weight.detach().clone()

        lrp(mutating_lrp_model, inputs)

        torch.testing.assert_close(mutating_lrp_model[0].weight, original)

    def test_preserves_fixed_context(self, lrp_context):
        model, batch = lrp_context
        context = batch["context"]
        previous = context.clone()

        result = lrp(model, batch, input_keys="values")

        assert torch.isfinite(result.attributions).all()
        torch.testing.assert_close(context, previous)
        assert not context.requires_grad
