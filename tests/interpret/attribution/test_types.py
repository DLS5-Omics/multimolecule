from multimolecule.interpret import ScalarTarget
from multimolecule.interpret.attribution import AttributionTarget


def test_attribution_target_aliases_scalar_target():
    target = ScalarTarget(class_idx=1, token_index=2)

    assert isinstance(target, AttributionTarget)
    assert target.class_idx == 1
    assert target.token_index == 2
