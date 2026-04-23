import torch

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret.attribution import FeatureAblationAttributor


def test_feature_ablation_supports_token_feature_masks(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    attributor = FeatureAblationAttributor(model)
    feature_mask = torch.tensor([[0, 1, 2, 3]])

    output = attributor(input_ids, baseline="mask", feature_mask=feature_mask)

    assert output.attributions.shape == (1, 4, 16)
    assert output.delta is None
    assert torch.allclose(output.attributions[0, :, 0], output.attributions[0, :, -1])
