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

from multimolecule import RnaFmConfig, RnaFmForSequencePrediction, RnaTokenizer
from multimolecule.interpret.attribution import feature_ablation, integrated_gradients, layer_integrated_gradients


@pytest.mark.parametrize("output", ["scores", lambda result: result["scores"]], ids=["field", "callable"])
def test_attribution_explains_selected_non_logit_output(output):
    pytest.importorskip("captum")

    class Prediction(nn.Module):
        def forward(self, inputs):
            sums = inputs.sum(dim=-1)
            return {"logits": 2 * sums, "scores": 3 * sums}

    result = integrated_gradients(Prediction(), torch.tensor([[1.0, -2.0]]), output=output, baselines=0, n_steps=4)

    torch.testing.assert_close(result.attributions, torch.tensor([[3.0, -6.0]]))


def test_integrated_gradients_explains_selected_output_for_tuple_inputs():
    pytest.importorskip("captum")

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


def test_feature_ablation_explains_integer_inputs():
    pytest.importorskip("captum")

    class Prediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weights", torch.tensor([0.0, 3.0, -2.0]))

        def forward(self, ids):
            return self.weights[ids].sum(dim=-1)

    result = feature_ablation(Prediction(), torch.tensor([[1, 2, 1]]), baselines=0, perturbations_per_eval=2)

    torch.testing.assert_close(result.attributions, torch.tensor([[3.0, -2.0, 3.0]]))
    assert result.convergence_delta is None


@pytest.mark.filterwarnings("ignore:Multiple layers provided:UserWarning")
def test_layer_integrated_gradients_supports_parallel_layers():
    pytest.importorskip("captum")

    class Prediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Identity()
            self.right = nn.Identity()

        def forward(self, inputs):
            return self.left(inputs[:, :1])[:, 0] + 2 * self.right(inputs[:, 1:])[:, 0]

    model = Prediction()
    result = layer_integrated_gradients(model, torch.tensor([[1.0, 2.0]]), layer=[model.left, "right"], n_steps=4)

    torch.testing.assert_close(result.attributions, [torch.tensor([[1.0]]), torch.tensor([[4.0]])])


def test_layer_integrated_gradients_rejects_duplicate_module_references():
    pytest.importorskip("captum")
    model = nn.Sequential(nn.Identity())

    with pytest.raises(ValueError, match="different module"):
        layer_integrated_gradients(model, torch.tensor([[1.0]]), layer=[model[0], "0"])


def _rnafm():
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


def test_rnafm_pad_baseline_keeps_input_positions_and_explains_target_logit():
    pytest.importorskip("captum")
    tokenizer = RnaTokenizer()
    tokens = tokenizer(["ACGU", "CG"], padding=True, return_tensors="pt", return_special_tokens_mask=True)
    input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
    baseline = input_ids.masked_fill(~tokens.special_tokens_mask.bool(), tokenizer.pad_token_id)
    model = _rnafm()

    positions = []
    with model.model.embeddings.position_embeddings.register_forward_pre_hook(
        lambda _module, args: positions.append(args[0].clone())
    ):
        with torch.no_grad():
            original = model(input_ids, attention_mask=attention_mask).logits[:, 1]
            model(baseline, attention_mask=attention_mask)
            original_positions, baseline_positions = positions
            assert not torch.equal(baseline_positions, original_positions)
            reference = model(baseline, attention_mask=attention_mask, position_ids=original_positions).logits[:, 1]
        positions.clear()
        result = layer_integrated_gradients(
            model,
            input_ids,
            layer="model.embeddings.word_embeddings",
            baselines=baseline,
            additional_forward_args=(attention_mask, original_positions),
            target=1,
            n_steps=32,
            internal_batch_size=8,
            return_convergence_delta=True,
        )

    assert positions
    for position_ids in positions:
        torch.testing.assert_close(position_ids, original_positions.repeat(len(position_ids) // len(input_ids), 1))
    assert result.attributions.shape == (*input_ids.shape, model.config.hidden_size)
    scores = result.attributions.sum(dim=-1)
    assert result.convergence_delta.shape == (len(input_ids),)
    excluded_scores = scores[tokens.special_tokens_mask.bool()]
    torch.testing.assert_close(excluded_scores, torch.zeros_like(excluded_scores))
    torch.testing.assert_close(scores.sum(dim=-1), original - reference, atol=1e-6, rtol=1e-3)
    torch.testing.assert_close(result.convergence_delta, torch.zeros(len(input_ids)), atol=1e-6, rtol=0)


def test_attribution_explains_how_to_install_missing_captum(monkeypatch):
    class MissingCaptum:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == "captum":
                raise ModuleNotFoundError("No module named 'captum'", name=fullname)

    monkeypatch.delitem(sys.modules, "captum", raising=False)
    monkeypatch.delitem(sys.modules, "captum.attr", raising=False)
    monkeypatch.setattr(sys, "meta_path", [MissingCaptum(), *sys.meta_path])
    with pytest.raises(ImportError, match=r"multimolecule\[interpret\]"):
        integrated_gradients(nn.Identity(), torch.tensor([[1.0]]))
