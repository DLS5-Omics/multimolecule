import numpy as np
import pytest

from multimolecule.data.rna_secondary_structure import (
    _DOT_BRACKET_PAIR_TABLE,
    contact_map_to_pairs,
    dot_bracket_to_pairs,
    pairs_to_contact_map,
    pairs_to_dot_bracket,
    pseudoknot_nucleotides,
    pseudoknot_pairs,
)


def test_dot_bracket_to_pairs_basic_and_pseudoknot():
    result = dot_bracket_to_pairs("((.))")
    expected = np.array([(0, 4), (1, 3)])
    assert np.array_equal(result, expected)
    # pseudoknot example using different bracket types
    db = "((.[.))]"
    result_pk = dot_bracket_to_pairs(db)
    expected_pk = np.array([(0, 6), (1, 5), (3, 7)])
    assert np.array_equal(result_pk, expected_pk)


def test_dot_bracket_to_pairs_errors():
    with pytest.raises(ValueError):
        dot_bracket_to_pairs("((.)")  # unmatched
    with pytest.raises(ValueError):
        dot_bracket_to_pairs("(a)")  # invalid symbol


def test_pairs_to_contact_map_length_inference_and_bounds():
    pairs = np.array([(0, 4), (1, 3)])
    cm = pairs_to_contact_map(pairs, length=5)
    assert cm.dtype == np.bool_ and cm.shape == (5, 5)
    assert cm.astype(int).tolist() == [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]

    # inference when length is None
    cm2 = pairs_to_contact_map(np.array([(2, 5)]))
    assert cm2.shape == (6, 6)

    # out of bounds
    with pytest.raises(ValueError):
        pairs_to_contact_map(np.array([(0, 10)]), length=5)


def test_contact_map_to_pairs_validations_and_unsafe():
    cm = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    result = contact_map_to_pairs(cm)
    expected = np.array([(0, 4), (1, 3)])
    assert np.array_equal(result, expected)

    # non-symmetric -> unsafe path symmetrizes from upper triangle
    cm_ns = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.warns(UserWarning):
        pairs = contact_map_to_pairs(cm_ns, unsafe=True)
    # after symmetrization and conflict resolution, (0,1) and (0,2) can both appear, but
    # greedy uniqueness per index will keep only one; ensure at least one pair kept
    assert len(pairs) > 0 and (pairs[0].tolist() == [0, 1] or pairs[0].tolist() == [0, 2])

    # diagonal non-zero -> unsafe clears diagonal
    cm_d = np.array([[1, 0], [0, 0]])
    with pytest.warns(UserWarning):
        result_d = contact_map_to_pairs(cm_d, unsafe=True)
    assert len(result_d) == 0

    # multiple pairings -> unsafe keeps first
    cm_mp = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    with pytest.warns(UserWarning):
        mp_pairs = contact_map_to_pairs(cm_mp, unsafe=True)
    assert len(mp_pairs) == 1


def test_pairs_to_dot_bracket_nested_crossing_and_errors():
    assert pairs_to_dot_bracket(np.array([(0, 3), (1, 2)]), length=4) == "(())"
    assert pairs_to_dot_bracket(np.array([(0, 2), (1, 3)]), length=4) == "([)]"

    # out of bounds strict
    with pytest.raises(ValueError):
        pairs_to_dot_bracket(np.array([(0, 5)]), length=4)

    # duplicates -> unsafe drops
    with pytest.warns(UserWarning):
        s = pairs_to_dot_bracket(np.array([(0, 3), (0, 2)]), length=4, unsafe=True)
    assert s.count("(") == 1 and s.count(")") == 1

    # too many tiers
    m = 1 + len(_DOT_BRACKET_PAIR_TABLE)
    pairs = np.array([(i, i + m) for i in range(m)])
    with pytest.raises(ValueError):
        pairs_to_dot_bracket(pairs, length=2 * m)


def test_pseudoknot_detection_pairs_and_nts():
    # nested => no pk
    assert pseudoknot_pairs(np.array([(0, 3), (1, 2)])).shape == (0, 2)
    assert pseudoknot_nucleotides(np.array([(0, 3), (1, 2)])).size == 0

    # crossing => both pairs and all nts are pk
    pkp = pseudoknot_pairs(np.array([(0, 2), (1, 3)]))
    assert np.array_equal(pkp, np.array([(0, 2), (1, 3)]))
    assert np.array_equal(pseudoknot_nucleotides(np.array([(0, 2), (1, 3)])), np.array([0, 1, 2, 3]))
