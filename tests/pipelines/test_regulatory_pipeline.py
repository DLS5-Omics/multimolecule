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
    AutoModelForRegulatoryActivityPrediction,
    AutoModelForRegulatoryProfilePrediction,
    AutoModelForRegulatoryTrackPrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    BasenjiBlockConfig,
    BasenjiConfig,
    BasenjiForTokenPrediction,
    BassetConfig,
    BassetForSequencePrediction,
    BpNetConfig,
    BpNetForProfilePrediction,
)
from multimolecule.pipelines.regulatory import (
    RegulatoryActivityPipeline,
    RegulatoryProfilePipeline,
    RegulatoryTrackPipeline,
    RegulatoryVariantEffectPipeline,
)


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

    def __init__(self, id2label=None):
        super().__init__()
        self.config = SimpleNamespace(id2label=id2label)

    def forward(self, input_ids=None, attention_mask=None):
        return {"logits": torch.ones(input_ids.size(0), 1)}


class _FeatureModel(_DummyModel):
    def __init__(self):
        super().__init__()
        self.features = None

    def forward(self, input_ids=None, attention_mask=None, features=None):
        self.features = features
        return {"logits": features.sum(dim=-1, keepdim=True)}


class _PostprocessModel(_DummyModel):
    output_channels = ["processed_a", "processed_b"]

    def __init__(self):
        super().__init__()
        self.postprocessed = False

    def postprocess(self, outputs):
        self.postprocessed = True
        return outputs["logits"] + 1


class _ProfileModel(_DummyModel):
    def __init__(self):
        super().__init__({0: "plus", 1: "minus"})
        self.postprocessed = False

    def postprocess(self, outputs):
        self.postprocessed = True
        return outputs["profile_logits"] + outputs["count_logits"].unsqueeze(1)


def _activity_pipeline(model=None) -> RegulatoryActivityPipeline:
    pipeline = object.__new__(RegulatoryActivityPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = model or _DummyModel({0: "cell_a", 1: "cell_b"})
    pipeline.task = "regulatory-activity"
    return pipeline


def _track_pipeline() -> RegulatoryTrackPipeline:
    pipeline = object.__new__(RegulatoryTrackPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = _DummyModel({0: "track_a", 1: "track_b"})
    pipeline.task = "regulatory-track"
    return pipeline


def _profile_pipeline() -> RegulatoryProfilePipeline:
    pipeline = object.__new__(RegulatoryProfilePipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = _ProfileModel()
    pipeline.task = "regulatory-profile"
    return pipeline


def _variant_pipeline(model=None) -> RegulatoryVariantEffectPipeline:
    pipeline = object.__new__(RegulatoryVariantEffectPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = model or _DummyModel()
    pipeline.task = "regulatory-variant-effect"
    return pipeline


def test_regulatory_activity_pipeline_reports_sequence_level_scores():
    pipeline = _activity_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[1.0, 2.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "ACG"
    assert result["channels"] == ["cell_a", "cell_b"]
    assert result["scores"] == {"cell_a": pytest.approx(1.0), "cell_b": pytest.approx(2.0)}


def test_regulatory_activity_pipeline_uses_model_postprocess_and_channels():
    model = _PostprocessModel()
    pipeline = _activity_pipeline(model)
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[1.0, 2.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert model.postprocessed
    assert result["channels"] == ["processed_a", "processed_b"]
    assert result["scores"] == {"processed_a": pytest.approx(2.0), "processed_b": pytest.approx(3.0)}


def test_regulatory_activity_pipeline_passes_features_to_feature_models_only():
    feature_model = _FeatureModel()
    pipeline = _activity_pipeline(feature_model)
    model_inputs = pipeline.preprocess("AC", features=[1.0, 2.0])

    result = pipeline._forward(model_inputs)

    assert feature_model.features.shape == (1, 2)
    assert result["logits"].item() == pytest.approx(3.0)

    plain_model = _DummyModel()
    pipeline = _activity_pipeline(plain_model)
    result = pipeline._forward(pipeline.preprocess("AC", features=[1.0, 2.0]))

    assert result["logits"].shape == (1, 1)


def test_regulatory_track_pipeline_reports_binned_scores():
    pipeline = _track_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["sequence"] == "ACG"
    assert result["bin_index_base"] == 0
    assert result["channels"] == ["track_a", "track_b"]
    assert result["tracks"] == [
        {"bin": 0, "track_a": pytest.approx(0.1), "track_b": pytest.approx(0.2)},
        {"bin": 1, "track_a": pytest.approx(0.3), "track_b": pytest.approx(0.4)},
    ]


def test_regulatory_track_pipeline_preserves_single_bin_channels():
    pipeline = _track_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1]], dtype=torch.long),
        "logits": torch.tensor([[[0.1, 0.2]]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["tracks"] == [{"bin": 0, "track_a": pytest.approx(0.1), "track_b": pytest.approx(0.2)}]


def test_regulatory_profile_pipeline_uses_model_postprocess():
    pipeline = _profile_pipeline()
    model_outputs = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "profile_logits": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        "count_logits": torch.tensor([[1.0, 2.0]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert pipeline.model.postprocessed
    assert result["sequence"] == "AC"
    assert result["position_index_base"] == 0
    assert result["channels"] == ["plus", "minus"]
    assert result["profile"] == [
        {"position": 0, "nucleotide": "A", "plus": pytest.approx(1.1), "minus": pytest.approx(2.2)},
        {"position": 1, "nucleotide": "C", "plus": pytest.approx(1.3), "minus": pytest.approx(2.4)},
    ]


def test_regulatory_variant_effect_pipeline_reports_separate_model_deltas():
    pipeline = _variant_pipeline()
    model_outputs = {
        "reference_sequence": "ACG",
        "alternative_sequence": "ATG",
        "reference_outputs": {"logits": torch.tensor([[1.0]])},
        "alternative_outputs": {"logits": torch.tensor([[1.75]])},
    }

    result = pipeline.postprocess(model_outputs)

    assert result["reference_sequence"] == "ACG"
    assert result["alternative_sequence"] == "ATG"
    assert result["channels"] == ["score"]
    assert result["reference_score"] == pytest.approx(1.0)
    assert result["alternative_score"] == pytest.approx(1.75)
    assert result["delta_score"] == pytest.approx(0.75)
    assert result["variant_effects"] == [
        {
            "channel": "score",
            "delta_score": pytest.approx(0.75),
            "direction": "gain",
            "reference_score": pytest.approx(1.0),
            "alternative_score": pytest.approx(1.75),
        }
    ]


def test_regulatory_variant_effect_pipeline_reports_ranked_channel_deltas():
    pipeline = _variant_pipeline(_DummyModel({0: "K562", 1: "HepG2"}))
    model_outputs = {
        "reference_sequence": "ACG",
        "alternative_sequence": "ATG",
        "reference_outputs": {"logits": torch.tensor([[1.0, 4.0]])},
        "alternative_outputs": {"logits": torch.tensor([[3.5, 3.0]])},
    }

    result = pipeline.postprocess(model_outputs, top_k=1)

    assert result["delta_scores"] == {"K562": pytest.approx(2.5), "HepG2": pytest.approx(-1.0)}
    assert result["variant_effects"] == [
        {
            "channel": "K562",
            "delta_score": pytest.approx(2.5),
            "direction": "gain",
            "reference_score": pytest.approx(1.0),
            "alternative_score": pytest.approx(3.5),
        }
    ]


def test_regulatory_variant_effect_pipeline_preserves_binned_axis_semantics():
    model = _DummyModel({0: "track"})
    model.config.pool_factor = 128
    pipeline = _variant_pipeline(model)
    model_outputs = {
        "reference_sequence": "A" * 256,
        "alternative_sequence": "C" * 256,
        "reference_outputs": {"logits": torch.tensor([[[1.0], [4.0]]])},
        "alternative_outputs": {"logits": torch.tensor([[[3.0], [1.0]]])},
    }

    result = pipeline.postprocess(model_outputs, top_k=1)

    assert result["bin_index_base"] == 0
    assert result["delta_scores"] == [
        {"bin": 0, "track": pytest.approx(2.0)},
        {"bin": 1, "track": pytest.approx(-3.0)},
    ]
    assert result["variant_effects"] == [
        {
            "bin": 1,
            "channel": "track",
            "delta_score": pytest.approx(-3.0),
            "direction": "loss",
            "reference_score": pytest.approx(4.0),
            "alternative_score": pytest.approx(1.0),
        }
    ]


def test_regulatory_variant_effect_pipeline_rejects_missing_or_mismatched_alternative():
    pipeline = _variant_pipeline()

    with pytest.raises(ValueError, match="alternative"):
        pipeline.preprocess("ACG")
    with pytest.raises(ValueError, match="same length"):
        pipeline.preprocess("ACG", alternative="AC")


def test_regulatory_pipelines_are_registered():
    assert PIPELINE_REGISTRY.check_task("regulatory-activity")[0] == "regulatory-activity"
    assert PIPELINE_REGISTRY.check_task("regulatory-track")[0] == "regulatory-track"
    assert PIPELINE_REGISTRY.check_task("regulatory-profile")[0] == "regulatory-profile"
    assert PIPELINE_REGISTRY.check_task("regulatory-variant-effect")[0] == "regulatory-variant-effect"


def test_regulatory_pipeline_defaults_are_transformers_compatible():
    activity_task = PIPELINE_REGISTRY.check_task("regulatory-activity")[1]
    track_task = PIPELINE_REGISTRY.check_task("regulatory-track")[1]
    profile_task = PIPELINE_REGISTRY.check_task("regulatory-profile")[1]
    variant_task = PIPELINE_REGISTRY.check_task("regulatory-variant-effect")[1]

    assert activity_task["default"]["model"] == ("multimolecule/basset", "main")
    assert track_task["default"]["model"] == ("multimolecule/enformer", "main")
    assert profile_task["default"]["model"] == ("multimolecule/bpnet", "main")
    assert variant_task["default"]["model"] == ("multimolecule/deepsea", "main")


def test_regulatory_auto_classes_keep_domain_specific_heads():
    basset_config = BassetConfig(
        sequence_length=16,
        num_conv_layers=1,
        conv_channels=[2],
        conv_kernel_sizes=[3],
        conv_pool_sizes=[1],
        fc_sizes=[4],
        num_labels=2,
    )
    basenji_config = BasenjiConfig(
        sequence_length=8,
        stem_channels=2,
        stem_pool_size=1,
        conv_tower_channels=[2],
        blocks=BasenjiBlockConfig(num_blocks=1, bottleneck_size=1),
        crop_bins=0,
        head_hidden_size=2,
        num_labels=2,
    )
    bpnet_config = BpNetConfig(hidden_size=2, num_dilated_layers=1, num_tasks=1, num_strands=1)

    assert isinstance(AutoModelForRegulatoryActivityPrediction.from_config(basset_config), BassetForSequencePrediction)
    assert isinstance(
        AutoModelForRegulatoryVariantEffectPrediction.from_config(basset_config), BassetForSequencePrediction
    )
    assert isinstance(AutoModelForRegulatoryTrackPrediction.from_config(basenji_config), BasenjiForTokenPrediction)
    assert isinstance(AutoModelForRegulatoryProfilePrediction.from_config(bpnet_config), BpNetForProfilePrediction)
