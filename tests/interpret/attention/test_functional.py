import torch

from multimolecule import CaLmForSequencePrediction, HyenaDnaForSequencePrediction
from multimolecule.interpret.attention import prepare_attention_output


def _forward_attentions(model, input_ids):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, return_dict=True)
    finally:
        model.train(was_training)
    return getattr(outputs, "attentions", None)


def test_prepare_attention_output_returns_raw_selected_attention_maps(calm_attention_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_attention_config)

    output = prepare_attention_output(_forward_attentions(model, input_ids), layers=[1], heads=[0, 2])

    assert output.attentions.shape == (1, 1, 2, 4, 4)
    assert output.layers == [1]
    assert output.heads == [0, 2]
    assert output.aggregation is None


def test_prepare_attention_output_supports_mean_aggregations(calm_attention_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_attention_config)
    attentions = _forward_attentions(model, input_ids)

    raw = prepare_attention_output(attentions)
    mean_heads = prepare_attention_output(attentions, aggregate="mean_heads")
    mean_layers = prepare_attention_output(attentions, aggregate="mean_layers")

    assert torch.allclose(mean_heads.attentions, raw.attentions.mean(dim=2))
    assert torch.allclose(mean_layers.attentions, raw.attentions.mean(dim=0))
    assert mean_heads.attentions.shape == (2, 1, 4, 4)
    assert mean_layers.attentions.shape == (1, 4, 4, 4)
    assert mean_heads.layers == [0, 1]
    assert mean_heads.heads is None
    assert mean_layers.layers is None
    assert mean_layers.heads == [0, 1, 2, 3]


def test_prepare_attention_output_supports_rollout(calm_attention_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_attention_config)

    output = prepare_attention_output(_forward_attentions(model, input_ids), aggregate="rollout")

    assert output.attentions.shape == (1, 4, 4)
    assert output.aggregation == "rollout"
    assert output.layers is None
    assert output.heads is None
    assert torch.allclose(
        output.attentions.sum(dim=-1),
        torch.ones_like(output.attentions.sum(dim=-1)),
        atol=1e-5,
    )


def test_prepare_attention_output_preserves_tokens(calm_attention_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_attention_config)

    output = prepare_attention_output(_forward_attentions(model, input_ids), tokens=["A", "C", "G", "U"])

    assert output.tokens == ["A", "C", "G", "U"]


def test_prepare_attention_output_fails_on_missing_attention_maps(hyenadna_config, input_ids):
    model = HyenaDnaForSequencePrediction(hyenadna_config)

    try:
        prepare_attention_output(_forward_attentions(model, input_ids))
    except ValueError as exc:
        assert "Attention maps are required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing attention maps")
