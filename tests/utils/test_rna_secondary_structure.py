import numpy as np
import pytest
import torch

from multimolecule.utils.rna_secondary_structure import (
    _DOT_BRACKET_PAIR_TABLE,
    contact_map_to_dot_bracket,
    contact_map_to_pairs,
    dot_bracket_to_contact_map,
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


def test_pairs_to_contact_map_numpy():
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


def test_pairs_to_contact_map_torch():
    pairs = torch.tensor([(0, 4), (1, 3)], dtype=torch.long)
    cm = pairs_to_contact_map(pairs, length=5)
    assert isinstance(cm, torch.Tensor) and cm.dtype == torch.bool and cm.shape == (5, 5)
    # self-pairing error
    with pytest.raises(ValueError):
        _ = pairs_to_contact_map(torch.tensor([(1, 1)], dtype=torch.long), length=2)


@pytest.mark.parametrize("backend", ["np", "torch"])
def test_pairs_contact_map_roundtrip_unique(backend: str):
    pairs = [(0, 5), (1, 4), (2, 3)]
    if backend == "np":
        cm = pairs_to_contact_map(np.array(pairs), length=6)
        back = contact_map_to_pairs(cm)
        back_list = {tuple(x) for x in back.tolist()}
    else:
        tcm = pairs_to_contact_map(torch.tensor(pairs, dtype=torch.long), length=6)
        back_t = contact_map_to_pairs(tcm)
        back_list = {tuple(map(int, p)) for p in back_t.tolist()}
        # device is preserved
        assert tcm.device.type == back_t.device.type
    assert back_list == {(min(i, j), max(i, j)) for i, j in pairs}


def test_contact_map_to_pairs_numpy():
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
    assert pairs.shape[1] == 2 and pairs.shape[0] >= 1

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
    assert mp_pairs.shape[0] == 1


def test_contact_map_to_pairs_torch():
    cm = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    result = contact_map_to_pairs(cm)
    assert isinstance(result, torch.Tensor) and result.dtype == torch.long
    assert result.shape == (2, 2)

    # invalid symmetry should raise without unsafe
    with pytest.raises(ValueError):
        _ = contact_map_to_pairs(torch.tensor([[0, 1], [0, 0]]), unsafe=False)


def test_contact_map_to_pairs_unsafe_resolves_multiples_numpy():
    # Build conflicting map with two neighbors for some rows
    cm = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )
    with pytest.warns(UserWarning):
        pairs = contact_map_to_pairs(cm, unsafe=True)
    # No node should be paired more than once
    used = set()
    for i, j in pairs.tolist():
        assert i not in used and j not in used
        used.add(i)
        used.add(j)


def test_pairs_to_dot_bracket_nested_crossing_and_errors_numpy():
    assert pairs_to_dot_bracket(np.array([(0, 3), (1, 2)]), length=4) == "(())"
    assert pairs_to_dot_bracket(np.array([(0, 2), (1, 3)]), length=4) == "([)]"

    # out of bounds strict
    with pytest.raises(ValueError):
        pairs_to_dot_bracket(np.array([(0, 5)]), length=4)

    # duplicates -> unsafe drops
    with pytest.warns(UserWarning):
        s = pairs_to_dot_bracket(np.array([(0, 3), (0, 2)]), length=4, unsafe=True)
    assert s.count("(") == 1 and s.count(")") == 1

    # too many tiers -> strict raises
    m = 1 + len(_DOT_BRACKET_PAIR_TABLE)
    pairs = np.array([(i, i + m) for i in range(m)])
    with pytest.raises(ValueError):
        pairs_to_dot_bracket(pairs, length=2 * m)
    # unsafe mode truncates and warns
    with pytest.warns(UserWarning):
        s = pairs_to_dot_bracket(pairs, length=2 * m, unsafe=True)
    assert len(s) == 2 * m


def test_pairs_to_dot_bracket_torch_accepts_tensor():
    s = pairs_to_dot_bracket(torch.tensor([(0, 3), (1, 2)], dtype=torch.long), length=4)
    assert s == "(())"


@pytest.mark.parametrize("s", ["())", "([)"])
def test_dot_bracket_to_pairs_unmatched_and_invalid_close(s):
    with pytest.raises(ValueError):
        _ = dot_bracket_to_pairs(s)


def test_dot_bracket_contact_map_roundtrip_numpy():
    s = "((.[.))]"
    cm = dot_bracket_to_contact_map(s)
    s2 = contact_map_to_dot_bracket(cm)
    assert len(s2) == len(s)
    # Roundtrip through pairs
    p = dot_bracket_to_pairs(s)
    s3 = pairs_to_dot_bracket(p, length=len(s))
    assert len(s3) == len(s)


def test_contact_map_to_dot_bracket_torch():
    # non-symmetric -> unsafe conversion to string
    cm = torch.tensor([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.warns(UserWarning):
        s = contact_map_to_dot_bracket(cm, unsafe=True)
    assert isinstance(s, str) and len(s) == 3


def test_pseudoknot_detection_pairs_and_nts_numpy():
    # nested => no pk
    assert pseudoknot_pairs(np.array([(0, 3), (1, 2)])).shape == (0, 2)
    assert pseudoknot_nucleotides(np.array([(0, 3), (1, 2)])).size == 0

    # crossing => both pairs and all nts are pk
    pkp = pseudoknot_pairs(np.array([(0, 2), (1, 3)]))
    assert np.array_equal(pkp, np.array([(0, 2), (1, 3)]))
    assert np.array_equal(pseudoknot_nucleotides(np.array([(0, 2), (1, 3)])), np.array([0, 1, 2, 3]))


def test_pseudoknot_detection_pairs_and_nts_torch():
    pkp = pseudoknot_pairs(torch.tensor([(0, 2), (1, 3)], dtype=torch.long))
    assert isinstance(pkp, torch.Tensor) and pkp.dtype == torch.long
    assert pkp.shape == (2, 2)
    nts = pseudoknot_nucleotides(torch.tensor([(0, 2), (1, 3)], dtype=torch.long))
    assert isinstance(nts, torch.Tensor) and nts.dtype == torch.long
    assert nts.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "pairs_np, expected_pk",
    [
        # two independent nested stems: (0,4)-(1,3) and (5,8)-(6,7)
        (np.array([(0, 4), (1, 3), (5, 8), (6, 7)]), np.empty((0, 2), dtype=int)),
        # simple crossing between (0,3) and (1,4)
        (np.array([(0, 3), (1, 4)]), np.array([(0, 3), (1, 4)])),
        # one nested (0,5)-(1,4) plus crossings with (2,7) => all three participate in at least one crossing
        (np.array([(0, 5), (2, 7), (1, 4)]), np.array([(0, 5), (2, 7), (1, 4)])),
    ],
)
def test_pseudoknot_pairs_various_numpy(pairs_np, expected_pk):
    out = pseudoknot_pairs(pairs_np)
    assert out.shape[1] == 2
    # Compare as sets to avoid ordering assumptions
    assert {tuple(x) for x in out.tolist()} == {tuple(x) for x in expected_pk.tolist()}
