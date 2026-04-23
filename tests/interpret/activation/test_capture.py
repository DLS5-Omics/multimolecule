import torch

from multimolecule import CaLmForSequencePrediction, HyenaDnaForSequencePrediction
from multimolecule.interpret import capture_activations


def test_capture_activations_supports_multiple_explicit_calm_layers(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    output = capture_activations(
        model,
        input_ids,
        ["model.embeddings.word_embeddings", "model.encoder.layer.0.output"],
    )

    assert output.requested_layers == ["model.embeddings.word_embeddings", "model.encoder.layer.0.output"]
    assert output.resolved_layers == ["model.embeddings.word_embeddings", "model.encoder.layer.0.output"]
    assert set(output.activations) == {"model.embeddings.word_embeddings", "model.encoder.layer.0.output"}
    assert output.activations["model.embeddings.word_embeddings"].shape == (1, 4, 16)
    assert output.activations["model.encoder.layer.0.output"].shape == (1, 4, 16)


def test_capture_activations_supports_semantic_selectors(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    output = capture_activations(
        model,
        input_ids,
        ["embeddings", "attention_outputs", "residual_outputs"],
    )

    assert output.requested_layers == ["embeddings", "attention_outputs", "residual_outputs"]
    assert output.resolved_layers == [
        "model.embeddings",
        "model.encoder.layer.0.attention.output",
        "model.encoder.layer.0",
    ]
    assert output.activations["model.embeddings"].shape == (1, 4, 16)
    assert output.activations["model.encoder.layer.0.attention.output"].shape == (1, 4, 16)
    assert output.activations["model.encoder.layer.0"].shape == (1, 4, 16)


def test_capture_activations_rejects_ambiguous_tuple_outputs(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    try:
        capture_activations(model, input_ids, "model.encoder.layer.0.attention")
    except TypeError as exc:
        assert "multiple tensors" in str(exc)
    else:
        raise AssertionError("Expected TypeError for ambiguous tuple-valued module outputs")


def test_capture_activations_supports_hyenadna(hyenadna_config, input_ids):
    torch.manual_seed(0)
    model = HyenaDnaForSequencePrediction(hyenadna_config)

    output = capture_activations(
        model,
        input_ids,
        ["model.embeddings.word_embeddings", "model.layers.0.mlp"],
    )

    assert output.activations["model.embeddings.word_embeddings"].shape == (1, 4, 16)
    assert output.activations["model.layers.0.mlp"].shape == (1, 4, 16)


def test_capture_activations_removes_hooks_after_repeated_calls(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    module = dict(model.named_modules())["model.encoder.layer.0.output"]

    assert len(module._forward_hooks) == 0
    capture_activations(model, input_ids, "model.encoder.layer.0.output")
    assert len(module._forward_hooks) == 0
    capture_activations(model, input_ids, "model.encoder.layer.0.output")
    assert len(module._forward_hooks) == 0
