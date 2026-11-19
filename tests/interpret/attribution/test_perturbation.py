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
    feature_ablation,
    feature_permutation,
    occlusion,
    shapley_value_sampling,
    shapley_values,
)

pytestmark = pytest.mark.usefixtures("captum")


class TestFeatureAblation:
    def test_attributes_integer_inputs(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weights", torch.tensor([0.0, 3.0, -2.0]))

            def forward(self, ids):
                return self.weights[ids].sum(dim=-1)

        result = feature_ablation(Prediction(), torch.tensor([[1, 2, 1]]), baselines=0, perturbations_per_eval=2)

        torch.testing.assert_close(result.attributions, torch.tensor([[3.0, -2.0, 3.0]]))
        assert result.convergence_delta is None

    def test_preserves_keyword_input_order(self, named_score_batch):
        model, batch = named_score_batch
        result = feature_ablation(
            model,
            batch,
            input_keys=("auxiliary", "values"),
            output="scores",
            baselines=(0, 0),
            perturbations_per_eval=2,
        )
        expected = (torch.tensor([[30.0], [36.0]]), torch.tensor([[2.0, 0.0], [0.0, 32.0]]))
        torch.testing.assert_close(result.attributions, expected)


class TestOcclusion:
    @pytest.mark.parametrize(
        ("stride", "expected"),
        [(1, [[3.0, 4.0, 6.0, 7.0], [7.0, 6.0, 4.0, 3.0]]), (2, [[3.0, 3.0, 7.0, 7.0], [7.0, 7.0, 3.0, 3.0]])],
        ids=["overlapping", "disjoint"],
    )
    def test_attributes_sliding_windows(self, stride, expected):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weights", torch.arange(5, dtype=torch.float32))

            def forward(self, ids, offset):
                score = self.weights[ids].sum(dim=-1) + offset
                return {"scores": torch.stack((-score, score), dim=-1)}

        result = occlusion(
            Prediction(),
            torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
            sliding_window_shapes=(2,),
            strides=stride,
            baselines=0,
            target=1,
            additional_forward_args=(torch.tensor([5.0, 7.0]),),
            output="scores",
            perturbations_per_eval=2,
        )

        torch.testing.assert_close(result.attributions, torch.tensor(expected))
        assert result.convergence_delta is None

    def test_accepts_scalar_features(self):
        result = occlusion(nn.Identity(), torch.tensor([1.0, -2.0]), sliding_window_shapes=(), baselines=0)

        torch.testing.assert_close(result.attributions, torch.tensor([1.0, -2.0]))


class TestShapleyValueSampling:
    def test_preserves_cross_input_groups(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weights", torch.arange(6, dtype=torch.float32))

            def forward(self, first, second):
                return {"scores": self.weights[first].sum(dim=-1) + self.weights[second].sum(dim=-1)}

        with torch.random.fork_rng(devices=[]):
            result = shapley_value_sampling(
                Prediction(),
                (torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])),
                baselines=(0, 0),
                output=lambda result: result["scores"],
                feature_mask=(torch.tensor([[0, 1, 0]]), torch.tensor([[1, 2]])),
                n_samples=5,
                perturbations_per_eval=2,
            )

        torch.testing.assert_close(result.attributions, (torch.tensor([[4.0, 6.0, 4.0]]), torch.tensor([[6.0, 5.0]])))
        assert result.convergence_delta is None


class TestShapleyValues:
    def test_splits_interactions_exactly(self):
        class Prediction(nn.Module):
            def forward(self, *, features, offset):
                return features[:, 0] * (1 + features[:, 1]) + offset

        batch = {"features": torch.tensor([[2.0, 3.0]]), "offset": torch.tensor([4.0])}
        result = shapley_values(Prediction(), batch, input_keys="features", perturbations_per_eval=2)

        torch.testing.assert_close(result.attributions, torch.tensor([[5.0, 3.0]]))


class TestFeaturePermutation:
    def test_attributes_grouped_inputs(self):
        class Prediction(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weights", torch.arange(9, dtype=torch.float32))

            def forward(self, ids, offset):
                score = self.weights[ids].sum(dim=-1) + offset
                return {"scores": torch.stack((2 * score, score), dim=-1)}

        with torch.random.fork_rng(devices=[]):
            result = feature_permutation(
                Prediction(),
                torch.tensor([[1, 2, 3], [6, 7, 8]]),
                target=1,
                additional_forward_args=(torch.tensor([3.0, 8.0]),),
                output="scores",
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=2,
            )

        torch.testing.assert_close(result.attributions, torch.tensor([[-10.0, -10.0, -5.0], [10.0, 10.0, 5.0]]))
        assert result.convergence_delta is None

    def test_accepts_custom_permutation(self):
        class Prediction(nn.Module):
            def forward(self, *, values, scale):
                return {"score": values.sum(dim=-1) * scale}

        def reverse_batch(values, mask):
            return torch.where(mask, values.flip(0), values)

        result = feature_permutation(
            Prediction(),
            {"values": torch.tensor([[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]]), "scale": torch.tensor([1.0, 0.5, 2.0])},
            input_keys="values",
            output=lambda result: result["score"],
            perm_func=reverse_batch,
            perturbations_per_eval=2,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([[-5.0, -5.0], [0.0, 0.0], [10.0, 10.0]]))

    def test_requires_multiple_examples(self):
        with pytest.raises(ValueError, match="at least two examples"):
            feature_permutation(nn.Identity(), torch.ones(1, 2), target=0)
