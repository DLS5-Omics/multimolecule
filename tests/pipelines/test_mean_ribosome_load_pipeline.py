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

from types import SimpleNamespace

import pytest
import torch
from transformers.pipelines import PIPELINE_REGISTRY

import multimolecule  # noqa: F401
from multimolecule import (
    AutoModelForMeanRibosomeLoadPrediction,
    BassetConfig,
    FramepoolConfig,
    FramepoolForSequencePrediction,
    MmSpliceConfig,
    Optimus5PrimeConfig,
    Optimus5PrimeForSequencePrediction,
    OptMrlConfig,
    OptMrlForSequencePrediction,
)
from multimolecule.pipelines.mean_ribosome_load import MeanRibosomeLoadPipeline


class _DummyTokenizer:
    tokens = "ACGTUN"

    def decode(self, input_ids, skip_special_tokens: bool = True):
        return "".join(self.tokens[int(index)] for index in input_ids)

    def __call__(self, sequence, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[self.tokens.index(base) for base in sequence]], dtype=torch.long),
            "attention_mask": torch.ones(1, len(sequence), dtype=torch.long),
        }


class _DummyModel(torch.nn.Module):
    base_model_prefix = "dummy"
    output_channels = ["mean_ribosome_load"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()

    def forward(self, input_ids=None, attention_mask=None):
        return {"logits": torch.ones(input_ids.size(0), 1)}


class _LibraryModel(_DummyModel):
    def __init__(self):
        super().__init__()
        self.library_indicator = None

    def forward(self, input_ids=None, attention_mask=None, library_indicator=None):
        self.library_indicator = library_indicator
        return {"logits": library_indicator.sum(dim=-1, keepdim=True)}


class _VectorModel(_DummyModel):
    output_channels = ["score_a", "score_b"]


def _mean_ribosome_load_pipeline(model=None) -> MeanRibosomeLoadPipeline:
    pipeline = object.__new__(MeanRibosomeLoadPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = model or _DummyModel()
    pipeline.task = "mean-ribosome-load"
    return pipeline


def test_mean_ribosome_load_pipeline_reports_scalar_score():
    pipeline = _mean_ribosome_load_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[1.25]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "ACG"
    assert result["channel"] == "mean_ribosome_load"
    assert result["score"] == pytest.approx(1.25)


def test_mean_ribosome_load_pipeline_reports_vector_scores():
    pipeline = _mean_ribosome_load_pipeline(_VectorModel())
    model_outputs = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "logits": torch.tensor([[1.0, 2.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "AC"
    assert result["channels"] == ["score_a", "score_b"]
    assert result["scores"] == {"score_a": pytest.approx(1.0), "score_b": pytest.approx(2.0)}


def test_mean_ribosome_load_pipeline_passes_supported_extra_inputs_only():
    model = _LibraryModel()
    pipeline = _mean_ribosome_load_pipeline(model)
    model_inputs = pipeline.preprocess(
        {"sequence": "AC", "library_indicator": [1.0, 2.0], "unused": [100.0]},
    )

    result = pipeline._forward(model_inputs)

    assert model.library_indicator.shape == (1, 2)
    assert result["logits"].item() == pytest.approx(3.0)


def test_mean_ribosome_load_pipeline_rejects_missing_sequence():
    pipeline = _mean_ribosome_load_pipeline()

    with pytest.raises(ValueError, match="sequence"):
        pipeline.preprocess({"library_indicator": [1.0]})


def test_mean_ribosome_load_pipeline_is_registered():
    assert PIPELINE_REGISTRY.check_task("mean-ribosome-load")[0] == "mean-ribosome-load"


def test_mean_ribosome_load_pipeline_default_is_transformers_compatible():
    task = PIPELINE_REGISTRY.check_task("mean-ribosome-load")[1]

    assert task["default"]["model"] == ("multimolecule/optimus5prime", "main")


def test_mean_ribosome_load_auto_class_keeps_domain_specific_heads():
    assert isinstance(
        AutoModelForMeanRibosomeLoadPrediction.from_config(Optimus5PrimeConfig()),
        Optimus5PrimeForSequencePrediction,
    )
    assert isinstance(AutoModelForMeanRibosomeLoadPrediction.from_config(OptMrlConfig()), OptMrlForSequencePrediction)
    assert isinstance(
        AutoModelForMeanRibosomeLoadPrediction.from_config(FramepoolConfig()), FramepoolForSequencePrediction
    )


@pytest.mark.parametrize("config", [BassetConfig(), MmSpliceConfig()])
def test_mean_ribosome_load_auto_class_rejects_non_mean_ribosome_load_models(config):
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForMeanRibosomeLoadPrediction.from_config(config)
