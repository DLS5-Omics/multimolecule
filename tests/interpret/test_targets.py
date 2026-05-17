from multimolecule.interpret import ScalarTarget


def test_scalar_target_fields():
    target = ScalarTarget(class_idx=1, token_index=2)

    assert target.class_idx == 1
    assert target.token_index == 2
