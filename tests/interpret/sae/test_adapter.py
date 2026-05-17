import torch
from torch import nn

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret import run_sae


class ToySae(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.encoder = nn.Linear(hidden_size, feature_size)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return self.encoder(activations)


class ToyCallableSae(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, feature_size)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.projection(activations)


def test_run_sae_supports_encode_protocol(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    sae = ToySae(hidden_size=16, feature_size=6)

    output = run_sae(sae, model, input_ids, layer="model.encoder.layer.0.output")

    assert output.features.shape == (1, 4, 6)
    assert output.feature_ids is None


def test_run_sae_supports_callable_modules(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    sae = ToyCallableSae(hidden_size=16, feature_size=5)
    feature_ids = torch.tensor([0, 2, 4, 6, 8])

    output = run_sae(
        sae,
        model,
        input_ids,
        layer="model.encoder.layer.0.output",
        feature_ids=feature_ids,
    )

    assert output.features.shape == (1, 4, 5)
    assert torch.equal(output.feature_ids, feature_ids)


def test_run_sae_supports_single_resolved_selector(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    sae = ToySae(hidden_size=16, feature_size=4)

    output = run_sae(sae, model, input_ids, layer="embeddings")

    assert output.features.shape == (1, 4, 4)


def test_run_sae_rejects_selectors_resolving_to_multiple_sources(calm_attention_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_attention_config)
    sae = ToySae(hidden_size=16, feature_size=4)

    try:
        run_sae(sae, model, input_ids, layer="residual_outputs")
    except ValueError as exc:
        assert "single activation source" in str(exc)
    else:
        raise AssertionError("Expected ValueError for selectors that resolve to multiple activation sources")
