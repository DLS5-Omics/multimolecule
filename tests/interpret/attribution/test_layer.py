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

from multimolecule import RnaFmConfig, RnaFmForSequencePrediction, RnaTokenizer
from multimolecule.interpret import attribution

pytestmark = pytest.mark.usefixtures("captum")


class LinearPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)
        self.readout = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
            self.readout.weight.copy_(torch.tensor([[1.0, 2.0]]))

    def forward(self, values, offset=0):
        score = self.readout(self.hidden(values)).squeeze(-1) + offset
        return {"scores": torch.stack((score, score * 2), dim=1)}


@pytest.fixture
def linear_prediction():
    return LinearPrediction()


class TestLayerIntegratedGradients:
    @pytest.mark.filterwarnings("ignore:Multiple layers provided:UserWarning")
    def test_supports_parallel_layers(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Identity()
                self.right = nn.Identity()

            def forward(self, inputs):
                return self.left(inputs[:, :1])[:, 0] + 2 * self.right(inputs[:, 1:])[:, 0]

        model = Prediction()
        result = attribution.layer_integrated_gradients(
            model, torch.tensor([[1.0, 2.0]]), layer=[model.left, "right"], n_steps=4
        )

        torch.testing.assert_close(result.attributions, [torch.tensor([[1.0]]), torch.tensor([[4.0]])])

    def test_rejects_duplicate_layers(self):
        model = nn.Sequential(nn.Identity())

        with pytest.raises(ValueError, match="different module"):
            attribution.layer_integrated_gradients(model, torch.tensor([[1.0]]), layer=[model[0], "0"])


@pytest.mark.parametrize("input_style", ["positional", "mapping"], scope="class")
class TestRnaFmLayerIntegratedGradients:
    @pytest.fixture(scope="class")
    def model(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            return RnaFmForSequencePrediction(
                RnaFmConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                    max_position_embeddings=16,
                    num_labels=2,
                )
            ).eval()

    @pytest.fixture(scope="class")
    def explanation(self, model, input_style):
        tokenizer = RnaTokenizer()
        batch = tokenizer(["ACGU", "CG"], padding=True, return_tensors="pt", return_special_tokens_mask=True)
        ids, mask = batch.input_ids, batch.attention_mask
        baseline = ids.masked_fill(~batch.special_tokens_mask.bool(), tokenizer.pad_token_id)
        # Explicit positions distinguish the expected baseline from one with inferred positions.
        positions = torch.arange(4, 4 + ids.size(1)).expand_as(ids)
        with torch.no_grad():
            original = model(ids, attention_mask=mask, position_ids=positions).logits[:, 1]
            reference = model(baseline, attention_mask=mask, position_ids=positions).logits[:, 1]
            inferred = model(baseline, attention_mask=mask).logits[:, 1]
        assert not torch.allclose(reference, inferred, atol=1e-6, rtol=1e-3)

        if input_style == "mapping":
            inputs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions}
            options = {"input_keys": "input_ids"}
        else:
            inputs = ids
            options = {"additional_forward_args": (mask, positions)}
        result = attribution.layer_integrated_gradients(
            model,
            inputs,
            layer=model.get_input_embeddings(),
            baselines=baseline,
            target=1,
            n_steps=32,
            internal_batch_size=8,
            return_convergence_delta=True,
            **options,
        )
        return result, batch, original - reference

    def test_explains_fixed_position_delta(self, explanation):
        result, _, expected = explanation

        torch.testing.assert_close(result.attributions.sum((1, 2)), expected, atol=1e-6, rtol=1e-3)
        torch.testing.assert_close(result.convergence_delta, torch.zeros_like(expected), atol=1e-6, rtol=0)

    def test_preserves_token_axes(self, explanation, model):
        result, batch, _ = explanation

        assert result.attributions.shape == (*batch.input_ids.shape, model.config.hidden_size)

    def test_excludes_unchanged_tokens(self, explanation):
        result, batch, _ = explanation
        excluded = result.attributions.sum(dim=-1)[batch.special_tokens_mask.bool()]

        torch.testing.assert_close(excluded, torch.zeros_like(excluded))


class TestLayerGradientXActivation:
    @pytest.mark.parametrize(
        ("to_input", "expected"),
        [(False, [[0.0, 14.0], [10.0, -18.0]]), (True, [[4.0, 10.0], [12.0, -20.0]])],
        ids=["output", "input"],
    )
    def test_selects_layer_side(self, to_input, expected):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.hidden.weight.copy_(torch.tensor([[2.0, 1.0], [-1.0, 3.0]]))

            def forward(self, inputs, offset):
                hidden = self.hidden(inputs)
                score = hidden[:, 0] - 2 * hidden[:, 1] + offset
                return {"scores": torch.stack((hidden[:, 0], score), dim=-1)}

        inputs = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        model = Prediction()
        result = attribution.layer_gradient_x_activation(
            model,
            inputs,
            layer="hidden",
            target=1,
            additional_forward_args=torch.tensor([3.0, -1.0]),
            output=lambda result: result["scores"],
            attribute_to_layer_input=to_input,
        )

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert result.convergence_delta is None

    def test_returns_unweighted_layer_list(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Identity()
                self.right = nn.Identity()

            def forward(self, inputs):
                return self.left(inputs[:, 0]) + 2 * self.right(inputs[:, 1])

        model = Prediction()
        result = attribution.layer_gradient_x_activation(
            model, torch.tensor([[3.0, -4.0]]), layer=[model.left, "right"], multiply_by_inputs=False
        )

        torch.testing.assert_close(result.attributions, [torch.tensor([1.0]), torch.tensor([2.0])])

    def test_preserves_argument_grad_flag(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Identity()

            def forward(self, inputs, factors):
                return (inputs * self.hidden(factors)).sum(dim=-1)

        model = Prediction()
        inputs = torch.tensor([[2.0, 3.0]])
        factors = torch.tensor([[4.0, -5.0]])

        result = attribution.layer_gradient_x_activation(
            model, inputs, layer=model.hidden, additional_forward_args=factors
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[8.0, -15.0]]))
        assert not factors.requires_grad


class TestLayerFeatureAblation:
    @pytest.mark.parametrize(
        ("target", "perturbations_per_eval", "second_example"),
        [(1, 2, [[4.0, 4.0], [-5.0, -5.0]]), ([1, 0], 1, [[0.0, 0.0], [0.0, 0.0]])],
        ids=["fixed-target", "per-example-targets"],
    )
    def test_ablates_grouped_activations(self, target, perturbations_per_eval, second_example):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding.from_pretrained(torch.tensor([[0.0, 0.0], [2.0, 4.0], [5.0, 1.0]]))

            def forward(self, ids, offset):
                hidden = self.embedding(ids)
                score = (hidden[..., 0] - 2 * hidden[..., 1]).sum(dim=-1) + offset
                return {"scores": torch.stack((offset, score), dim=-1)}

        model = Prediction()
        result = attribution.layer_feature_ablation(
            model,
            torch.tensor([[1, 2], [2, 1]]),
            layer=model.embedding,
            layer_baselines=1,
            target=target,
            additional_forward_args=torch.tensor([3.0, -1.0]),
            output="scores",
            layer_mask=torch.tensor([[[0, 0], [1, 1]]]),
            perturbations_per_eval=perturbations_per_eval,
        )

        expected = torch.tensor([[[-5.0, -5.0], [4.0, 4.0]], second_example])
        torch.testing.assert_close(result.attributions, expected)
        assert result.convergence_delta is None


class TestLayerActivation:
    def test_captures_multiple_layers(self):
        class SignatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.ReLU()
                self.right = nn.Sigmoid()

            def forward(self, *, values, scale):
                self.left(values * scale)
                self.right(values)
                return {"metadata": "opaque"}

        model = SignatureModel()
        values = torch.tensor([[-3.0, 4.0]])
        result = attribution.layer_activation(
            model, {"values": values, "scale": 2.0}, layer=[model.left, "right"], input_keys="values"
        )

        torch.testing.assert_close(result.attributions, [torch.tensor([[0.0, 8.0]]), values.sigmoid()])


class TestLayerScoreAttribution:
    @pytest.mark.parametrize(
        ("attribute", "options", "expected", "has_delta"),
        [
            (
                attribution.layer_conductance,
                {"n_steps": 4, "method": "riemann_trapezoid"},
                [[4.0, 24.0], [12.0, 48.0]],
                True,
            ),
            (attribution.internal_influence, {"n_steps": 4}, [[2.0, 4.0], [2.0, 4.0]], False),
            (attribution.layer_deep_lift, {}, [[4.0, 24.0], [12.0, 48.0]], True),
            (attribution.layer_deep_lift_shap, {"baselines": torch.zeros(2, 2)}, [[4.0, 24.0], [12.0, 48.0]], True),
            (
                attribution.layer_gradient_shap,
                {"baselines": torch.zeros(2, 2), "n_samples": 2},
                [[4.0, 24.0], [12.0, 48.0]],
                True,
            ),
        ],
        ids=["conductance", "internal-influence", "deep-lift", "deep-lift-shap", "gradient-shap"],
    )
    def test_explains_affine_score(self, linear_prediction, attribute, options, expected, has_delta):
        kwargs = {"baselines": 0, **options}
        if has_delta:
            kwargs["return_convergence_delta"] = True

        result = attribute(
            linear_prediction,
            {"values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "offset": torch.tensor([0.5, 0.25])},
            layer="hidden",
            input_keys="values",
            target=1,
            output=lambda result: result["scores"],
            **kwargs,
        )

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        if has_delta:
            torch.testing.assert_close(result.convergence_delta, torch.zeros_like(result.convergence_delta))
        else:
            assert result.convergence_delta is None


class TestLayerGradCam:
    def test_preserves_signed_contributions(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
                with torch.no_grad():
                    self.conv.weight.copy_(torch.tensor([[[[2.0]]], [[[3.0]]]]))

            def forward(self, inputs):
                hidden = self.conv(inputs)
                return (hidden[:, 0] - hidden[:, 1]).sum(dim=(1, 2))

        inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        result = attribution.layer_grad_cam(Prediction(), inputs, layer="conv")

        torch.testing.assert_close(result.attributions, -inputs)


class TestLayerFeaturePermutation:
    @pytest.mark.parametrize(
        ("target", "last_example"),
        [
            (1, [8.0, 24.0]),
            (torch.tensor(1), [8.0, 24.0]),
            ([1, 0], [4.0, 12.0]),
            (torch.tensor([1, 0], dtype=torch.int32), [4.0, 12.0]),
        ],
        ids=["class-index", "scalar-tensor", "per-example-list", "per-example-tensor"],
    )
    def test_attributes_selected_classes(self, linear_prediction, target, last_example):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = attribution.layer_feature_permutation(
            linear_prediction, inputs, layer="hidden", target=target, output="scores", perturbations_per_eval=2
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[-8.0, -24.0], last_example]))

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ((1, 0), [[-12.0, -36.0], [12.0, 36.0]]),
            ([(0, 1), (1, 0)], [[-8.0, -24.0], [12.0, 36.0]]),
        ],
        ids=["fixed-coordinates", "per-example-coordinates"],
    )
    def test_selects_coordinate_targets(self, linear_prediction, target, expected):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        def output(result):
            scores = result["scores"]
            return torch.stack((scores, 3 * scores), dim=1)

        result = attribution.layer_feature_permutation(
            linear_prediction,
            inputs,
            layer="hidden",
            target=target,
            output=output,
            perturbations_per_eval=2,
        )

        torch.testing.assert_close(result.attributions, torch.tensor(expected))

    def test_rejects_floating_targets(self, linear_prediction):
        with pytest.raises(TypeError, match="integer"):
            attribution.layer_feature_permutation(
                linear_prediction,
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                layer="hidden",
                target=torch.tensor([1.0, 0.0]),
                output="scores",
            )


class TestLayerDeepLift:
    @pytest.fixture
    def model(self):
        class Prediction(LinearPrediction):
            def __init__(self):
                super().__init__()
                self.first = nn.ReLU()

            def forward(self, values, offset=0):
                return super().forward(self.first(values), offset)

        return Prediction()

    def test_explains_selected_score(self, model):
        inputs = torch.tensor([[1.0, 2.0]])
        result = attribution.layer_deep_lift(model, inputs, layer="hidden", output="scores", target=0)

        torch.testing.assert_close(result.attributions, torch.tensor([[2.0, 12.0]]))

    def test_preserves_caller_state(self, model):
        original = object()
        model.first.input = original
        inputs = torch.tensor([[1.0, 2.0]])

        attribution.layer_deep_lift(model, inputs, layer="hidden", output="scores", target=0)

        assert not inputs.requires_grad
        assert model.first.input is original


class TestLayerLrp:
    def test_reports_each_layer(self, linear_prediction):
        model = linear_prediction
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = attribution.layer_lrp(
            model, inputs, layer=[model.hidden, "readout"], target=0, output="scores", return_convergence_delta=True
        )

        expected = [torch.tensor([[2.0, 12.0], [6.0, 24.0]]), torch.tensor([[14.0], [30.0]])]
        torch.testing.assert_close(result.attributions, expected)
        torch.testing.assert_close(result.convergence_delta, [torch.zeros(2), torch.zeros(2)])

    def test_preserves_caller_attribute(self, linear_prediction):
        original = {"previous": object()}
        linear_prediction.hidden.activations = original

        attribution.layer_lrp(linear_prediction, torch.tensor([[1.0, 2.0]]), layer="hidden", target=0, output="scores")

        assert linear_prediction.hidden.activations is original

    def test_restores_weights(self, mutating_lrp_model):
        inputs = torch.tensor([[1.0, 2.0]])
        original = mutating_lrp_model[0].weight.detach().clone()

        attribution.layer_lrp(mutating_lrp_model, inputs, layer=mutating_lrp_model[0])

        torch.testing.assert_close(mutating_lrp_model[0].weight, original)

    def test_preserves_fixed_context(self, lrp_context):
        model, batch = lrp_context
        context = batch["context"]
        previous = context.clone()

        result = attribution.layer_lrp(model, batch, layer=model.primary, input_keys="values")

        assert torch.isfinite(result.attributions).all()
        torch.testing.assert_close(context, previous)
        assert not context.requires_grad
