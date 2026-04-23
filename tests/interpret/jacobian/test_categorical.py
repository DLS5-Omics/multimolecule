import torch

from multimolecule import CaLmForSequencePrediction, CaLmForTokenPrediction
from multimolecule.interpret import ScalarTarget, categorical_jacobian


def test_categorical_jacobian_returns_full_vocab_scores_for_sequence_targets(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    output = categorical_jacobian(model, input_ids)

    assert output.scores.shape == (1, 4, 8)
    assert output.scores.dtype == torch.float32
    assert output.positions is None
    assert output.top_k_indices is None


def test_categorical_jacobian_supports_token_targets_and_position_filtering(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForTokenPrediction(calm_config)
    target = ScalarTarget(token_index=1)

    output = categorical_jacobian(model, input_ids, target=target, positions=[0, 2])

    assert output.scores.shape == (1, 2, 8)
    assert output.positions is not None
    assert torch.equal(output.positions, torch.tensor([0, 2]))


def test_categorical_jacobian_supports_top_k_and_abs_reduction(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    output = categorical_jacobian(model, input_ids, top_k=3, reduction="abs")

    assert output.reduction == "abs"
    assert output.top_k_indices is not None
    assert output.top_k_scores is not None
    assert output.top_k_indices.shape == (1, 4, 3)
    assert output.top_k_scores.shape == (1, 4, 3)
    assert torch.allclose(output.top_k_scores, output.scores.gather(dim=-1, index=output.top_k_indices))


def test_categorical_jacobian_rejects_contact_like_outputs(calm_config, input_ids):
    class ContactModel(CaLmForSequencePrediction):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            output.logits = output.logits[:, None, None, :]
            return output

    model = ContactModel(calm_config)

    try:
        categorical_jacobian(model, input_ids)
    except NotImplementedError as exc:
        assert "not implemented yet" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for contact-like outputs")
