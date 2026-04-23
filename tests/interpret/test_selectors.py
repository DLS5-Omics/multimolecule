from multimolecule import CaLmForSequencePrediction, HyenaDnaForSequencePrediction
from multimolecule.interpret.selectors import resolve_modules


def test_resolve_modules_returns_requested_named_modules(calm_config):
    model = CaLmForSequencePrediction(calm_config)

    selection = resolve_modules(model, ["model.embeddings.word_embeddings", "model.encoder.layer.0.output"])

    assert selection.requested_layers == ["model.embeddings.word_embeddings", "model.encoder.layer.0.output"]
    assert selection.resolved_layers == ["model.embeddings.word_embeddings", "model.encoder.layer.0.output"]
    assert set(selection.modules) == {"model.embeddings.word_embeddings", "model.encoder.layer.0.output"}


def test_resolve_modules_supports_semantic_selectors(calm_attention_config):
    model = CaLmForSequencePrediction(calm_attention_config)

    selection = resolve_modules(model, ["embeddings", "attention_outputs", "residual_outputs"])

    assert selection.requested_layers == ["embeddings", "attention_outputs", "residual_outputs"]
    assert selection.resolved_layers == [
        "model.embeddings",
        "model.encoder.layer.0.attention.output",
        "model.encoder.layer.1.attention.output",
        "model.encoder.layer.0",
        "model.encoder.layer.1",
    ]
    assert set(selection.modules) == set(selection.resolved_layers)


def test_resolve_modules_rejects_unknown_module_paths(calm_config):
    model = CaLmForSequencePrediction(calm_config)

    try:
        resolve_modules(model, "model.does_not_exist")
    except ValueError as exc:
        assert "Unknown layer selector" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown layer path")


def test_resolve_modules_rejects_selectors_without_matches(hyenadna_config):
    model = HyenaDnaForSequencePrediction(hyenadna_config)

    try:
        resolve_modules(model, "attention_outputs")
    except ValueError as exc:
        assert "did not match any modules" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unmatched selector")
