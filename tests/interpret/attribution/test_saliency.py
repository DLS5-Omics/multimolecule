import torch

from multimolecule import CaLmForSequencePrediction, HyenaDnaForSequencePrediction
from multimolecule.interpret import attribute
from multimolecule.interpret.attribution import SaliencyAttributor


def test_saliency_supports_signed_gradients(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    attributor = SaliencyAttributor(model)

    output = attributor(input_ids, abs=False)

    assert output.attributions.shape == (1, 4, 16)
    assert output.token_attributions.shape == (1, 4)
    assert output.delta is None


def test_saliency_smoke_on_hyenadna_sequence_model(hyenadna_config, input_ids):
    torch.manual_seed(0)
    model = HyenaDnaForSequencePrediction(hyenadna_config)

    output = attribute(model, input_ids, method="saliency")

    assert output.attributions.shape == (1, 4, 16)
    assert output.delta is None
