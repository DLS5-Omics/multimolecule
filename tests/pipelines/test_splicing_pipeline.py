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
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY

import multimolecule  # noqa: F401
from multimolecule import (
    AutoModelForSpliceSitePrediction,
    AutoModelForSpliceVariantEffectPrediction,
    DnaTokenizer,
    MaxEntScanConfig,
    MaxEntScanModel,
    OpenSpliceAiConfig,
    OpenSpliceAiForTokenPrediction,
    PangolinConfig,
    PangolinModel,
    PangolinStageConfig,
    SpTransformerConfig,
    SpTransformerFeatureEncoderConfig,
    SpTransformerModel,
)
from multimolecule.pipelines.splicing import SpliceSitePipeline, SpliceVariantEffectPipeline


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

    def __init__(self, model_type: str):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)


class _DummyPangolinModel(_DummyModel):
    def __init__(self):
        super().__init__("pangolin")
        self.config.tissue_names = ["heart"]
        self.received_full_outputs = False

    def postprocess(self, outputs):
        self.received_full_outputs = "logits" in outputs and "input_ids" in outputs
        return outputs["logits"], ["heart_no_splice", "heart_splice_site", "heart_usage"]


def _site_pipeline(model_type: str = "openspliceai") -> SpliceSitePipeline:
    pipeline = object.__new__(SpliceSitePipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = _DummyPangolinModel() if model_type == "pangolin" else _DummyModel(model_type)
    pipeline.task = "splice-site"
    pipeline.threshold = 0.5
    pipeline.output_scores = True
    pipeline.top_k = None
    return pipeline


def _variant_pipeline(model_type: str = "maxentscan") -> SpliceVariantEffectPipeline:
    pipeline = object.__new__(SpliceVariantEffectPipeline)
    pipeline.tokenizer = _DummyTokenizer()
    pipeline.model = _DummyModel(model_type)
    pipeline.task = "splice-variant-effect"
    return pipeline


def test_splice_site_pipeline_decodes_acceptor_and_donor_probabilities():
    pipeline = _site_pipeline("openspliceai")
    model_outputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "logits": torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]]),
    }

    result = pipeline.postprocess(model_outputs, threshold=0.8)

    assert result["sequence"] == "ACG"
    assert result["channels"] == ["no_splice", "acceptor", "donor"]
    assert [(site["position"], site["nucleotide"], site["type"]) for site in result["splice_sites"]] == [
        (1, "C", "acceptor"),
        (2, "G", "donor"),
    ]
    assert [site["score"] for site in result["splice_sites"]] == pytest.approx([0.9867033, 0.9867033])
    assert len(result["scores"]) == 3


def test_splice_site_pipeline_supports_pangolin_tissue_channels():
    pipeline = _site_pipeline("pangolin")
    pipeline.model.config.num_tissues = 1
    model_outputs = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "logits": torch.tensor([[[0.1, 0.9, 0.4], [0.7, 0.3, 0.2]]]),
    }

    result = pipeline.postprocess(model_outputs, threshold=0.5)

    assert pipeline.model.received_full_outputs
    assert result["channels"] == ["heart_no_splice", "heart_splice_site", "heart_usage"]
    assert result["splice_sites"] == [
        {"position": 0, "nucleotide": "A", "type": "heart_splice_site", "score": pytest.approx(0.9)}
    ]


def test_splice_variant_effect_pipeline_reports_separate_model_deltas():
    pipeline = _variant_pipeline("maxentscan")
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


def test_splice_variant_effect_pipeline_reports_native_delta_outputs():
    pipeline = _variant_pipeline("mmsplice")
    model_outputs = {
        "reference_sequence": "AAAA",
        "alternative_sequence": "AAAT",
        "logits": torch.tensor([[0.25]]),
    }

    result = pipeline.postprocess(model_outputs)

    assert result["channels"] == ["score"]
    assert result["delta_score"] == pytest.approx(0.25)
    assert "reference_score" not in result


def test_splice_variant_effect_pipeline_reports_top_position_deltas():
    pipeline = _variant_pipeline("openspliceai")
    model_outputs = {
        "reference_sequence": "ACG",
        "alternative_sequence": "ATG",
        "reference_outputs": {"logits": torch.tensor([[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])},
        "alternative_outputs": {"logits": torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]])},
    }

    result = pipeline.postprocess(model_outputs, top_k=1)

    assert result["position_index_base"] == 0
    assert result["variant_effects"] == [
        {
            "position": 1,
            "reference_nucleotide": "C",
            "alternative_nucleotide": "T",
            "channel": "acceptor",
            "delta_score": pytest.approx(0.9800549),
            "direction": "gain",
            "reference_score": pytest.approx(0.00664835),
            "alternative_score": pytest.approx(0.9867033),
        }
    ]


def test_splice_variant_effect_pipeline_rejects_missing_or_mismatched_alternative():
    pipeline = _variant_pipeline("maxentscan")

    with pytest.raises(ValueError, match="alternative"):
        pipeline.preprocess("ACG")
    with pytest.raises(ValueError, match="same length"):
        pipeline.preprocess("ACG", alternative="AC")


def test_splicing_pipelines_are_registered():
    assert PIPELINE_REGISTRY.check_task("splice-site")[0] == "splice-site"
    assert PIPELINE_REGISTRY.check_task("splice-variant-effect")[0] == "splice-variant-effect"


def test_splicing_pipeline_defaults_are_transformers_compatible():
    site_task = PIPELINE_REGISTRY.check_task("splice-site")[1]
    variant_task = PIPELINE_REGISTRY.check_task("splice-variant-effect")[1]

    assert site_task["default"]["model"] == ("multimolecule/openspliceai-mane-400nt", "main")
    assert variant_task["default"]["model"] == ("multimolecule/mmsplice", "main")


def test_splicing_auto_classes_keep_native_splice_predictors():
    pangolin_stage = PangolinStageConfig(num_blocks=1, kernel_size=3, dilation=1)
    pangolin_config = PangolinConfig(context=4, num_tissues=1, num_ensemble=1, stages=[pangolin_stage])
    sptransformer_encoder = SpTransformerFeatureEncoderConfig(hidden_size=4)
    sptransformer_config = SpTransformerConfig(
        context=2,
        hidden_size=8,
        encoders=[sptransformer_encoder],
        attention_hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_local_attention_heads=1,
        intermediate_size=32,
        bucket_size=4,
        max_seq_len=8,
        num_tissues=2,
    )

    assert isinstance(AutoModelForSpliceSitePrediction.from_config(pangolin_config), PangolinModel)
    assert isinstance(AutoModelForSpliceVariantEffectPrediction.from_config(pangolin_config), PangolinModel)
    assert isinstance(AutoModelForSpliceSitePrediction.from_config(sptransformer_config), SpTransformerModel)
    assert isinstance(AutoModelForSpliceVariantEffectPrediction.from_config(sptransformer_config), SpTransformerModel)


def test_splicing_models_define_postprocess_channel_semantics():
    openspliceai = OpenSpliceAiForTokenPrediction(OpenSpliceAiConfig())
    scores, channels = openspliceai.postprocess(torch.tensor([[[5.0, 0.0, 0.0]]]))

    assert channels == ["no_splice", "acceptor", "donor"]
    assert scores.sum(dim=-1).item() == pytest.approx(1.0)

    sptransformer_encoder = SpTransformerFeatureEncoderConfig(hidden_size=4)
    sptransformer = SpTransformerModel(
        SpTransformerConfig(
            context=2,
            hidden_size=8,
            encoders=[sptransformer_encoder],
            attention_hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_local_attention_heads=1,
            intermediate_size=32,
            bucket_size=4,
            max_seq_len=8,
            num_tissues=2,
            tissue_names=["brain", "heart"],
        )
    )
    scores, channels = sptransformer.postprocess(torch.tensor([[[5.0, 0.0, 0.0, 0.25, 0.5]]]))

    assert channels == ["no_splice", "acceptor", "donor", "brain", "heart"]
    assert scores[..., :3].sum(dim=-1).item() == pytest.approx(1.0)
    assert scores[..., 3:].tolist() == [[[0.25, 0.5]]]


def test_splicing_pipeline_runs_parameter_free_maxentscan_model():
    model = MaxEntScanModel(MaxEntScanConfig())
    tokenizer = DnaTokenizer(["A", "C", "G", "T", "N"], unk_token="N", pad_token="N")

    predictor = pipeline("splice-site", model=model, tokenizer=tokenizer)
    result = predictor("CAGGTAAGT")

    assert result["sequence"] == "CAGGTAAGT"
    assert result["channels"] == ["score"]
    assert isinstance(result["score"], float)
