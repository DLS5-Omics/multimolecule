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
    gradient_shap,
    input_x_gradient,
    integrated_gradients,
    saliency,
    smoothgrad,
    smoothgrad_sq,
    vargrad,
)


@pytest.fixture
def linear():
    class Linear(nn.Module):
        def forward(self, inputs, scale=1):
            score = (-2 * inputs[:, 0] + 3 * inputs[:, 1]) * scale
            return {"logits": torch.stack((inputs.sum(dim=-1), score), dim=-1)}

    return Linear()


@pytest.fixture
def quadratic():
    class Quadratic(nn.Module):
        def forward(self, inputs):
            return inputs.square().sum(dim=-1)

    return Quadratic()


class TestIntegratedGradients:
    @pytest.mark.usefixtures("captum")
    @pytest.mark.parametrize("output", ["scores", lambda result: result["scores"]], ids=["field", "callable"])
    def test_attributes_selected_output(self, output):
        class Prediction(nn.Module):
            def forward(self, inputs):
                sums = inputs.sum(dim=-1)
                return {"logits": 2 * sums, "scores": 3 * sums}

        result = integrated_gradients(Prediction(), torch.tensor([[1.0, -2.0]]), output=output, baselines=0, n_steps=4)

        torch.testing.assert_close(result.attributions, torch.tensor([[3.0, -6.0]]))

    @pytest.mark.usefixtures("captum")
    def test_attributes_tuple_inputs(self):
        class Prediction(nn.Module):
            def forward(self, first, second, offset):
                sums = first.sum(dim=-1)
                return {"logits": torch.stack((sums, 2 * sums + 3 * second[:, 0] + offset), dim=-1)}

        result = integrated_gradients(
            Prediction(),
            (torch.tensor([[1.0, -2.0], [3.0, 4.0]]), torch.tensor([[2.0], [5.0]])),
            baselines=(0, 0),
            target=1,
            additional_forward_args=(torch.tensor([3.0, -1.0]),),
            method="riemann_middle",
            n_steps=2,
            return_convergence_delta=True,
        )

        expected = (torch.tensor([[2.0, -4.0], [6.0, 8.0]]), torch.tensor([[6.0], [15.0]]))
        torch.testing.assert_close(result.attributions, expected)
        torch.testing.assert_close(result.convergence_delta, torch.zeros(2))

    @pytest.mark.usefixtures("captum")
    def test_preserves_keyword_input_order(self, named_score_batch):
        model, batch = named_score_batch
        result = integrated_gradients(
            model,
            batch,
            input_keys=("auxiliary", "values"),
            output="scores",
            baselines=(0, 0),
            n_steps=4,
            internal_batch_size=4,
        )
        expected = (torch.tensor([[30.0], [36.0]]), torch.tensor([[2.0, 0.0], [0.0, 32.0]]))
        torch.testing.assert_close(result.attributions, expected)

    @pytest.mark.parametrize(
        ("input_keys", "error", "message"),
        [
            (None, ValueError, "input_keys"),
            ((), ValueError, "at least one"),
            (("values", "values"), ValueError, "unique"),
            ("missing", ValueError, "missing"),
            ("scale", TypeError, "tensor"),
        ],
        ids=["missing-keys", "empty", "duplicate", "unknown", "nontensor"],
    )
    def test_rejects_invalid_input_keys(self, input_keys, error, message):
        with pytest.raises(error, match=message):
            integrated_gradients(nn.Identity(), {"values": torch.ones(1, 1), "scale": 2}, input_keys=input_keys)

    def test_requires_shared_batch_dimension(self):
        batch = {"first": torch.ones(2, 1), "second": torch.ones(1, 1)}

        with pytest.raises(ValueError, match="same nonempty batch dimension"):
            integrated_gradients(nn.Identity(), batch, input_keys=("first", "second"))

    def test_reports_missing_captum(self, without_captum):
        result = without_captum(
            """
            import torch
            from multimolecule.interpret.attribution import integrated_gradients

            try:
                integrated_gradients(torch.nn.Identity(), torch.tensor([[1.0]]))
            except ImportError as error:
                assert "multimolecule[interpret]" in str(error)
            else:
                raise AssertionError("Expected Captum to be required for attribution.")
            """
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("captum")
class TestSaliency:
    @pytest.mark.parametrize(
        ("absolute", "expected"), [(True, [[4.0, 6.0]]), (False, [[-4.0, 6.0]])], ids=["absolute", "signed"]
    )
    def test_controls_gradient_sign(self, linear, absolute, expected):
        result = saliency(
            linear, torch.tensor([[1.0, 2.0]]), target=1, abs=absolute, additional_forward_args=torch.tensor([2.0])
        )

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert result.convergence_delta is None


@pytest.mark.usefixtures("captum")
class TestInputXGradient:
    def test_multiplies_gradients_by_inputs(self, linear):
        result = input_x_gradient(linear, torch.tensor([[1.0, 2.0], [-3.0, 4.0]]), target=1)

        torch.testing.assert_close(result.attributions, torch.tensor([[-2.0, 6.0], [6.0, 12.0]]))
        assert result.convergence_delta is None


@pytest.mark.usefixtures("captum")
class TestGradientState:
    @pytest.mark.parametrize("method", [saliency, input_x_gradient])
    def test_restores_grad_flag_on_error(self, method):
        class FailingModel(nn.Module):
            def forward(self, inputs):
                raise RuntimeError("forward failed")

        inputs = torch.tensor([[1.0, 2.0]])

        with pytest.raises(RuntimeError, match="forward failed"):
            method(FailingModel(), inputs)

        assert not inputs.requires_grad


@pytest.mark.usefixtures("captum")
class TestGradientShap:
    def test_accepts_callable_baselines(self, linear):
        def baseline_distribution(inputs):
            return inputs.new_zeros((3, inputs.size(-1)))

        result = gradient_shap(
            linear,
            torch.tensor([[1.0, 2.0], [-3.0, 4.0]]),
            baselines=baseline_distribution,
            target=1,
            n_samples=3,
            return_convergence_delta=True,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[-2.0, 6.0], [6.0, 12.0]]))
        torch.testing.assert_close(result.convergence_delta, torch.zeros(6))


@pytest.mark.usefixtures("captum")
class TestGradientStatistics:
    def test_averages_saliency(self, quadratic):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            result = smoothgrad(quadratic, torch.zeros((1, 2)), nt_samples=64, nt_samples_batch_size=8, stdevs=0.5)

        # A single saliency evaluation at zero would give zero, whereas nearby gradients have positive magnitude.
        torch.testing.assert_close(result.attributions, torch.full((1, 2), 0.8), atol=0.35, rtol=0)
        assert result.convergence_delta is None

    def test_averages_squared_gradients(self, quadratic):
        result = smoothgrad_sq(
            quadratic, torch.tensor([[1.0, -2.0]], requires_grad=True), nt_samples=3, stdevs=0.0, abs=False
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[4.0, 16.0]]))

    def test_measures_gradient_variation(self, quadratic):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            result = vargrad(
                quadratic,
                torch.zeros(1, 2),
                nt_samples=128,
                nt_samples_batch_size=16,
                stdevs=0.5,
                abs=False,
            )

        torch.testing.assert_close(result.attributions, torch.ones(1, 2), atol=0.35, rtol=0)
