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

import sys

import pytest
import torch
from torch import nn

from multimolecule.interpret.attribution import kernel_shap, lime, lime_base

pytestmark = pytest.mark.usefixtures("captum")


@pytest.fixture
def sklearn():
    pytest.importorskip("sklearn")


class TestKernelShap:
    @pytest.mark.usefixtures("sklearn")
    @pytest.mark.parametrize(
        ("return_input_shape", "expected"),
        [(True, [[-2.0, 21.0, 21.0]]), (False, [[-2.0, 21.0]])],
        ids=["input-shape", "coefficients"],
    )
    def test_explains_grouped_features(self, return_input_shape, expected):
        class Prediction(nn.Module):
            def forward(self, *, values, offset):
                score = -2 * values[:, 0] + 3 * values[:, 1] + 5 * values[:, 2] + offset
                return {"scores": torch.stack((values.sum(dim=-1) + offset, score), dim=-1)}

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            result = kernel_shap(
                Prediction(),
                {"values": torch.tensor([[1.0, 2.0, 3.0]]), "offset": torch.tensor([5.0])},
                input_keys=("values",),
                output="scores",
                target=1,
                baselines=0,
                feature_mask=torch.tensor([[0, 1, 1]]),
                n_samples=12,
                perturbations_per_eval=4,
                return_input_shape=return_input_shape,
            )

        torch.testing.assert_close(result.attributions, torch.tensor(expected), rtol=2e-4, atol=2e-4)
        assert result.convergence_delta is None

    @pytest.mark.parametrize(
        ("missing_name", "error_type", "match"),
        [
            ("sklearn", ImportError, r"scikit-learn.*multimolecule\[interpret\]"),
            ("missing_transitive_module", ModuleNotFoundError, "Backend import failed"),
        ],
        ids=["missing-backend", "missing-dependency"],
    )
    def test_reports_backend_import_errors(self, monkeypatch, missing_name, error_type, match):
        class UnavailableSklearn:
            @staticmethod
            def find_spec(fullname, path=None, target=None):
                if fullname == "sklearn":
                    raise ModuleNotFoundError("Backend import failed", name=missing_name)

        monkeypatch.delitem(sys.modules, "sklearn", raising=False)
        monkeypatch.setattr(sys, "meta_path", [UnavailableSklearn(), *sys.meta_path])
        with pytest.raises(error_type, match=match):
            kernel_shap(nn.Identity(), torch.tensor([[1.0, 2.0]]), target=0)


@pytest.mark.usefixtures("sklearn")
class TestLime:
    def test_fits_selected_score(self):
        class Prediction(nn.Module):
            def forward(self, features):
                score = 2 * features[:, 0] - 3 * features[:, 1]
                return {"scores": torch.stack((features.sum(dim=1), score), dim=1)}

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            result = lime(
                Prediction(),
                torch.tensor([[1.0, 2.0]]),
                output="scores",
                target=1,
                n_samples=128,
                perturbations_per_eval=8,
            )

        torch.testing.assert_close(result.attributions, torch.tensor([[2.0, -6.0]]), atol=0.2, rtol=0.0)


class TestLimeBase:
    def test_fits_supplied_samples(self):
        class Surrogate:
            def fit(self, data):
                features, scores, weights = next(iter(data))
                roots = weights.sqrt()
                self.coefficients = torch.linalg.lstsq(features * roots[:, None], scores * roots).solution

            def representation(self):
                return self.coefficients

        class Prediction(nn.Module):
            def forward(self, *, features, multiplier):
                return (3 * features[:, 0] + 5 * features[:, 1]) * multiplier

        def samples(inputs, **kwargs):
            yield from (inputs.new_tensor([row]) for row in [(0, 0), (1, 0), (0, 1), (1, 1)])

        result = lime_base(
            Prediction(),
            {"features": torch.tensor([[2.0, -1.0]]), "multiplier": 2},
            input_keys="features",
            interpretable_model=Surrogate(),
            similarity_func=lambda *args, **kwargs: 1.0,
            perturb_func=samples,
            perturb_interpretable_space=True,
            from_interp_rep_transform=lambda sample, inputs, **kwargs: sample * inputs,
            n_samples=4,
            perturbations_per_eval=2,
        )

        torch.testing.assert_close(result.attributions, torch.tensor([12.0, -10.0]))
