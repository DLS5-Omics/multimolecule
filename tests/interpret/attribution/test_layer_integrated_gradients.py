import torch

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret.attribution import LayerIntegratedGradientsAttributor


def test_layer_integrated_gradients_sequence_attribution_shapes(
    calm_config,
    padded_input_ids,
    padded_attention_mask,
):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    attributor = LayerIntegratedGradientsAttributor(model)

    output = attributor(
        padded_input_ids,
        baseline="pad",
        attention_mask=padded_attention_mask,
    )

    assert output.attributions.shape == (1, 6, 16)
    assert output.token_attributions.shape == (1, 6)
    assert output.delta is not None
    assert torch.equal(output.token_attributions[:, 4:], torch.zeros(1, 2))
