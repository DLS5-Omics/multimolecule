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
    A2zChromatinConfig,
    A2zChromatinForSequencePrediction,
    AutoModelForMethylationPrediction,
    BassetConfig,
    DeepCpgDnaConfig,
    DeepCpgDnaForSequencePrediction,
    MmSpliceConfig,
)
from multimolecule.pipelines.methylation import MethylationPipeline


class _DummyTokenizer:
    tokens = "ACGTN"

    def decode(self, input_ids, skip_special_tokens: bool = True):
        return "".join(self.tokens[int(index)] for index in input_ids)

    def __call__(self, sequence, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[self.tokens.index(base) for base in sequence]], dtype=torch.long),
            "attention_mask": torch.ones(1, len(sequence), dtype=torch.long),
        }


class _DummyModel(torch.nn.Module):
    base_model_prefix = "dummy"
    output_channels = ["cell_a", "cell_b"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()

    def forward(self, input_ids=None, attention_mask=None):
        return {"logits": torch.zeros(input_ids.size(0), 2)}

    def postprocess(self, outputs):
        return torch.sigmoid(outputs["logits"])


def _methylation_pipeline() -> MethylationPipeline:
    pipeline = object.__new__(MethylationPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = _DummyModel()
    pipeline.task = "methylation"
    return pipeline


def test_methylation_pipeline_reports_sequence_level_scores():
    pipeline = _methylation_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[0.0, 2.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "ACG"
    assert result["channels"] == ["cell_a", "cell_b"]
    assert result["scores"] == {"cell_a": pytest.approx(0.5), "cell_b": pytest.approx(0.880797)}


def test_methylation_pipeline_rejects_missing_sequence():
    pipeline = _methylation_pipeline()

    with pytest.raises(ValueError, match="sequence"):
        pipeline.preprocess({})


def test_methylation_pipeline_is_registered():
    assert PIPELINE_REGISTRY.check_task("methylation")[0] == "methylation"


def test_methylation_pipeline_default_is_transformers_compatible():
    task = PIPELINE_REGISTRY.check_task("methylation")[1]

    assert task["default"]["model"] == ("multimolecule/deepcpgdna", "main")


def test_methylation_auto_class_keeps_domain_specific_heads():
    assert isinstance(
        AutoModelForMethylationPrediction.from_config(DeepCpgDnaConfig()), DeepCpgDnaForSequencePrediction
    )
    assert isinstance(
        AutoModelForMethylationPrediction.from_config(A2zChromatinConfig()), A2zChromatinForSequencePrediction
    )


@pytest.mark.parametrize("config", [BassetConfig(), MmSpliceConfig()])
def test_methylation_auto_class_rejects_non_methylation_models(config):
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForMethylationPrediction.from_config(config)
