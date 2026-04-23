import torch

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret.attribution import IntegratedGradientsAttributor


def test_integrated_gradients_supports_zero_embedding_baseline(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    attributor = IntegratedGradientsAttributor(model)

    output = attributor(input_ids, baseline="zero", n_steps=8)

    assert output.attributions.shape == (1, 4, 16)
    assert output.delta is not None
