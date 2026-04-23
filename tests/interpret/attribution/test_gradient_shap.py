import torch

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret import attribute


def test_gradient_shap_runs_through_convenience_api(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)

    output = attribute(model, input_ids, method="gradient_shap", baseline="zero", n_samples=2, stdevs=0.0)

    assert output.attributions.shape == (1, 4, 16)
    assert output.delta is not None
