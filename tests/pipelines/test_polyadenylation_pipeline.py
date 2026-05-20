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
    Aparent2Config,
    Aparent2ForSequencePrediction,
    AparentConfig,
    AparentForSequencePrediction,
    AutoModelForPolyadenylationPrediction,
    BassetConfig,
    MmSpliceConfig,
)
from multimolecule.pipelines.polyadenylation import PolyadenylationPipeline


class _DummyTokenizer:
    tokens = "ACGTN"

    def decode(self, input_ids, skip_special_tokens: bool = True):
        return "".join(self.tokens[int(index)] for index in input_ids)

    def __call__(self, sequence, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[self.tokens.index(base) for base in sequence]], dtype=torch.long),
            "attention_mask": torch.ones(1, len(sequence), dtype=torch.long),
        }


class _CleavageModel(torch.nn.Module):
    base_model_prefix = "dummy"
    output_channels = ["cleavage_0", "cleavage_1", "no_cleavage"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()

    def forward(self, input_ids=None, attention_mask=None):
        return {"logits": torch.zeros(input_ids.size(0), 3)}

    def postprocess(self, outputs):
        return torch.softmax(outputs["logits"], dim=-1)


class _IsoformModel(_CleavageModel):
    output_channels = ["isoform_proportion"]

    def forward(self, input_ids=None, attention_mask=None):
        return {"logits": torch.zeros(input_ids.size(0), 1)}

    def postprocess(self, outputs):
        return torch.sigmoid(outputs["logits"])


def _polyadenylation_pipeline(model=None) -> PolyadenylationPipeline:
    pipeline = object.__new__(PolyadenylationPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = model or _CleavageModel()
    pipeline.task = "polyadenylation"
    return pipeline


def test_polyadenylation_pipeline_reports_cleavage_distribution():
    pipeline = _polyadenylation_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[0.0, 1.0, 2.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "ACG"
    assert result["cleavage_distribution"] == [
        {"position": 0, "probability": pytest.approx(0.0900306)},
        {"position": 1, "probability": pytest.approx(0.2447285)},
        {"event": "no_cleavage", "probability": pytest.approx(0.665241)},
    ]


def test_polyadenylation_pipeline_reports_isoform_score():
    pipeline = _polyadenylation_pipeline(_IsoformModel())
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[0.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "ACG"
    assert result["channel"] == "isoform_proportion"
    assert result["score"] == pytest.approx(0.5)


def test_polyadenylation_pipeline_rejects_missing_sequence():
    pipeline = _polyadenylation_pipeline()

    with pytest.raises(ValueError, match="sequence"):
        pipeline.preprocess({})


def test_polyadenylation_pipeline_is_registered():
    assert PIPELINE_REGISTRY.check_task("polyadenylation")[0] == "polyadenylation"


def test_polyadenylation_pipeline_default_is_transformers_compatible():
    task = PIPELINE_REGISTRY.check_task("polyadenylation")[1]

    assert task["default"]["model"] == ("multimolecule/aparent2", "main")


def test_polyadenylation_auto_class_keeps_domain_specific_heads():
    assert isinstance(AutoModelForPolyadenylationPrediction.from_config(AparentConfig()), AparentForSequencePrediction)
    assert isinstance(
        AutoModelForPolyadenylationPrediction.from_config(Aparent2Config()), Aparent2ForSequencePrediction
    )


@pytest.mark.parametrize("config", [BassetConfig(), MmSpliceConfig()])
def test_polyadenylation_auto_class_rejects_non_polyadenylation_models(config):
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForPolyadenylationPrediction.from_config(config)
