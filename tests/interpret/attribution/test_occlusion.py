import torch

from multimolecule import CaLmForSequencePrediction
from multimolecule.interpret import attribute
from multimolecule.interpret.attribution import OcclusionAttributor


def test_occlusion_supports_mask_baseline_and_custom_ids(calm_config, input_ids):
    torch.manual_seed(0)
    model = CaLmForSequencePrediction(calm_config)
    attributor = OcclusionAttributor(model)
    custom_baseline = torch.full_like(input_ids, 4)
    output_via_helper = attribute(
        model,
        input_ids,
        method="occlusion",
        sliding_window_shapes=(1, 16),
        strides=(1, 16),
    )

    output_mask = attributor(input_ids, baseline="mask")
    output_custom = attributor(input_ids, baseline=custom_baseline)

    assert output_mask.attributions.shape == (1, 4, 16)
    assert output_custom.attributions.shape == (1, 4, 16)
    assert output_via_helper.attributions.shape == (1, 4, 16)
