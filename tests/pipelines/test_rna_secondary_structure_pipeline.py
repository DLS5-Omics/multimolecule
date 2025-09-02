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


import numpy as np
import torch

from multimolecule.pipelines.rna_secondary_structure import RnaSecondaryStructurePipeline


class _DummyTokenizer:
    def decode(self, input_ids, skip_special_tokens: bool = True):
        return "AAA"

    def batch_decode(self, input_ids, skip_special_tokens: bool = True):
        return ["AAA", "A"]


class _DummyModel:
    base_model_prefix = "dummy"


def _make_pipeline() -> RnaSecondaryStructurePipeline:
    pipeline = object.__new__(RnaSecondaryStructurePipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = _DummyModel()
    pipeline.threshold = 0.5
    pipeline.output_contact_map = False
    return pipeline


def test_postprocess_batch_respects_output_contact_map_override():
    pipeline = _make_pipeline()

    input_ids = torch.tensor([[1], [2]], dtype=torch.long)
    logits = torch.full((2, 3, 3), -10.0)
    logits[0, 0, 2] = 10.0
    logits[0, 2, 0] = 10.0
    model_outputs = {"input_ids": input_ids, "logits": logits}

    results = pipeline.postprocess(model_outputs, output_contact_map=True)
    assert isinstance(results, list) and len(results) == 2

    assert results[0]["sequence"] == "AAA"
    assert results[0]["secondary_structure"] == "(.)"
    assert isinstance(results[0]["contact_map"], np.ndarray)
    assert results[0]["contact_map"].shape == (3, 3)

    assert results[1]["sequence"] == "A"
    assert results[1]["secondary_structure"] == "."
    assert isinstance(results[1]["contact_map"], np.ndarray)
    assert results[1]["contact_map"].shape == (1, 1)

    results_no_cm = pipeline.postprocess(model_outputs, output_contact_map=False)
    assert "contact_map" not in results_no_cm[0]
    assert "contact_map" not in results_no_cm[1]


def test_postprocess_single_returns_dict():
    pipeline = _make_pipeline()

    input_ids = torch.tensor([[1]], dtype=torch.long)
    logits = torch.full((1, 3, 3), -10.0)
    logits[0, 0, 2] = 10.0
    logits[0, 2, 0] = 10.0
    model_outputs = {"input_ids": input_ids, "logits": logits}

    result = pipeline.postprocess(model_outputs, output_contact_map=True)
    assert isinstance(result, dict)
    assert result["sequence"] == "AAA"
    assert result["secondary_structure"] == "(.)"
    assert isinstance(result["contact_map"], np.ndarray)
    assert result["contact_map"].shape == (3, 3)
