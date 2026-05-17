import torch

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret import attribute


def test_attribute_supports_layer_integrated_gradients_zero_baseline(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    output = attribute(model, input_ids, method="layer_integrated_gradients", baseline="zero", n_steps=8)

    assert output.attributions.shape == (1, 4, 16)
    assert output.delta is not None


def test_attribute_rejects_contact_like_outputs(calm_config, input_ids):
    class ContactModel(CaLmForSequencePrediction):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            output.logits = output.logits[:, None, None, :]
            return output

    torch.manual_seed(0)
    model = ContactModel(calm_config)

    try:
        attribute(model, input_ids)
    except NotImplementedError as exc:
        assert "not implemented yet" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for contact-like outputs")
