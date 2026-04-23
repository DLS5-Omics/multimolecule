import torch

from multimolecule import CaLmForTokenPrediction
from multimolecule.interpret import ScalarTarget
from multimolecule.interpret.attribution import InputXGradientAttributor


def test_input_x_gradient_token_attribution(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForTokenPrediction(calm_config)
    attributor = InputXGradientAttributor(model)

    output = attributor(
        input_ids,
        target=ScalarTarget(token_index=1),
    )

    assert output.attributions.shape == (1, 4, 16)
    assert output.token_attributions.shape == (1, 4)
    assert output.delta is None
