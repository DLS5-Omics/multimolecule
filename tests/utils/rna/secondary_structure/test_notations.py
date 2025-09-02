# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


from __future__ import annotations

import string

import numpy as np
import pytest
import torch

from multimolecule.utils.rna.secondary_structure import notations
from tests.utils.rna.secondary_structure.conftest import (
    CROSSING_PAIRS,
    DOT_BRACKET_ERROR_CASES,
    as_list,
    as_numpy,
    as_set,
    as_tuple_list,
    make_contact_map,
    make_pairs,
)


def _count_openers(dot_bracket: str) -> set[str]:
    openers = set("([{<" + string.ascii_uppercase)
    return {ch for ch in dot_bracket if ch in openers}


def test_pairs_to_dot_bracket_minimal_tiers() -> None:
    pairs = np.array([(0, 2), (1, 6), (3, 5), (4, 7)])
    dot_bracket = notations.pairs_to_dot_bracket(pairs, length=8)

    assert len(_count_openers(dot_bracket)) == 2
    got = as_set(notations.dot_bracket_to_pairs(dot_bracket))
    expected = {(0, 2), (1, 6), (3, 5), (4, 7)}
    assert got == expected


def test_dot_bracket_to_pairs_valid() -> None:
    dot_bracket = "A+.(a)_"
    pairs = notations.dot_bracket_to_pairs(dot_bracket)
    assert as_list(pairs) == [[0, 4], [3, 5]]


def test_dot_bracket_to_pairs_empty() -> None:
    empty = notations.dot_bracket_to_pairs("...")
    assert empty.shape == (0, 2)


@pytest.mark.parametrize("dot_bracket, match", DOT_BRACKET_ERROR_CASES)
def test_dot_bracket_to_pairs_errors(dot_bracket: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        notations.dot_bracket_to_pairs(dot_bracket)


def test_notations_type_errors() -> None:
    with pytest.raises(TypeError):
        notations.pairs_to_contact_map(object())
    with pytest.raises(TypeError):
        notations.contact_map_to_pairs(object())
    torch_map = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.int)
    np_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int)
    assert notations.contact_map_to_pairs(torch_map, unsafe=True).shape[1] == 2
    assert len(notations.contact_map_to_pairs(np_map, unsafe=True)) > 0


def test_dot_bracket_contact_map_roundtrip() -> None:
    dot = "(())"
    contact_map = notations.dot_bracket_to_contact_map(dot)
    assert contact_map.dtype == bool
    assert notations.contact_map_to_dot_bracket(contact_map) == dot


def test_pairs_to_contact_map_basic(backend: str) -> None:
    pairs = make_pairs([(2, 0), (1, 3)], backend)
    contact_map = notations.pairs_to_contact_map(pairs)
    expected = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(as_numpy(contact_map), expected)


def test_pairs_to_contact_map_empty(backend: str) -> None:
    empty_pairs = make_pairs([], backend)
    empty_map = notations.pairs_to_contact_map(empty_pairs)
    assert as_numpy(empty_map).shape == (0, 0)


@pytest.mark.parametrize(
    "pairs",
    [np.array([1, 2, 3]), torch.tensor([1, 2, 3]), [1, 2, 3]],
    ids=["numpy", "torch", "list"],
)
def test_pairs_to_contact_map_shape_errors(pairs) -> None:
    with pytest.raises(ValueError, match="pairs must be"):
        notations.pairs_to_contact_map(pairs)


def test_pairs_to_contact_map_self_pairing_error(backend: str) -> None:
    pairs = make_pairs([[1, 1], [0, 2]], backend)
    with pytest.raises(ValueError, match="Self-pairing"):
        notations.pairs_to_contact_map(pairs, length=3)


def test_pairs_to_contact_map_self_pairing_unsafe(backend: str) -> None:
    pairs = make_pairs([[1, 1], [0, 2]], backend)
    with pytest.warns(UserWarning, match="Self-pairing"):
        contact_map = notations.pairs_to_contact_map(pairs, length=3, unsafe=True)
    assert int(as_numpy(contact_map).sum()) == 2


def test_pairs_to_contact_map_self_only(backend: str) -> None:
    pairs = make_pairs([[1, 1]], backend)
    with pytest.warns(UserWarning, match="Self-pairing"):
        contact_map = notations.pairs_to_contact_map(pairs, unsafe=True)
    contact_map_np = as_numpy(contact_map)
    assert contact_map_np.shape == (2, 2)
    assert int(contact_map_np.sum()) == 0


def test_pairs_to_contact_map_out_of_bounds(backend: str) -> None:
    pairs = make_pairs([[0, 3]], backend)
    with pytest.raises(ValueError, match="out of bounds"):
        notations.pairs_to_contact_map(pairs, length=3)


@pytest.mark.parametrize(
    "data, match",
    [
        pytest.param([[0, 1, 0], [1, 0, 0]], "square", id="not_square"),
        pytest.param([[0, 1], [0, 0]], "not symmetric", id="not_symmetric"),
        pytest.param([[1, 0], [0, 0]], "diagonal", id="diagonal"),
        pytest.param([[0, 1, 1], [1, 0, 0], [1, 0, 0]], "multiple", id="multiple_pairings"),
    ],
)
def test_contact_map_to_pairs_binary_errors(backend: str, data, match: str) -> None:
    contact_map = make_contact_map(data, backend, dtype=bool)
    with pytest.raises(ValueError, match=match):
        notations.contact_map_to_pairs(contact_map)


def test_contact_map_to_pairs_binary_valid(backend: str) -> None:
    contact_map = make_contact_map([[0, 1], [1, 0]], backend, dtype=bool)
    pairs = notations.contact_map_to_pairs(contact_map)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_binary_not_symmetric_unsafe(backend: str) -> None:
    contact_map = make_contact_map([[0, 1], [0, 0]], backend, dtype=bool)
    with pytest.warns(UserWarning, match="not symmetric"):
        pairs = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_binary_diagonal_unsafe(backend: str) -> None:
    contact_map = make_contact_map([[1, 0], [0, 0]], backend, dtype=bool)
    with pytest.warns(UserWarning, match="diagonal"):
        pairs = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(pairs) == []


def test_contact_map_to_pairs_binary_multiple_pairings_unsafe(backend: str) -> None:
    contact_map = make_contact_map([[0, 1, 1], [1, 0, 0], [1, 0, 0]], backend, dtype=bool)
    with pytest.warns(UserWarning, match="paired to multiple"):
        pairs = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_binary_empty(array_backend: str) -> None:
    if array_backend == "numpy":
        contact_map = np.zeros((0, 0), dtype=bool)
    else:
        contact_map = torch.zeros((0, 0), dtype=torch.bool)
    pairs = notations.contact_map_to_pairs(contact_map)
    assert pairs.shape == (0, 2)


def test_contact_map_to_pairs_float_basic(backend: str) -> None:
    contact_map = make_contact_map([[0.0, 0.9], [0.9, 0.0]], backend, dtype=torch.float)
    pairs = notations.contact_map_to_pairs(contact_map)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_float_threshold(backend: str) -> None:
    low_map = make_contact_map([[0.0, 0.4], [0.4, 0.0]], backend, dtype=torch.float)
    pairs = notations.contact_map_to_pairs(low_map, threshold=0.5)
    assert as_tuple_list(pairs) == []


def test_contact_map_to_pairs_float_tied_scores() -> None:
    contact_map = np.array(
        [
            [0.0, 0.9, 0.9],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    with pytest.warns(UserWarning, match="Multiple pairings"):
        pairs_np = notations.contact_map_to_pairs(contact_map, unsafe=True)
        pairs_pt = notations.contact_map_to_pairs(torch.tensor(contact_map), unsafe=True)
    assert as_tuple_list(pairs_np) == as_tuple_list(pairs_pt)


def test_contact_map_to_pairs_float_empty(array_backend: str) -> None:
    if array_backend == "numpy":
        contact_map = np.empty((0, 0), dtype=float)
    else:
        contact_map = torch.empty((0, 0), dtype=torch.float)
    pairs = notations.contact_map_to_pairs(contact_map)
    assert pairs.shape == (0, 2)


@pytest.mark.parametrize(
    "data, match, kwargs",
    [
        pytest.param([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "square", {}, id="not_square"),
        pytest.param([[0.0, 0.0], [0.0, 0.0]], "threshold", {"threshold": 1.5}, id="threshold"),
        pytest.param([[0.0, float("nan")], [0.0, 0.0]], "NaN", {}, id="nan"),
        pytest.param([[1.0, 0.8], [0.8, 0.0]], "Diagonal", {}, id="diagonal"),
        pytest.param(
            [[0.0, 0.9, 0.8], [0.9, 0.0, 0.0], [0.8, 0.0, 0.0]], "Multiple pairings", {}, id="multiple_pairings"
        ),
        pytest.param([[0.0, 2.0], [-2.0, 0.0]], "probabilities", {}, id="out_of_range"),
        pytest.param([[0.0, 0.9], [0.1, 0.0]], "not symmetric", {}, id="not_symmetric"),
        pytest.param([[0.0, 0.8], [0.0, 0.0]], "not symmetric", {}, id="upper_only"),
    ],
)
def test_contact_map_to_pairs_float_errors(backend: str, data, match: str, kwargs: dict) -> None:
    contact_map = make_contact_map(data, backend, dtype=float)
    with pytest.raises(ValueError, match=match):
        notations.contact_map_to_pairs(contact_map, **kwargs)


def test_contact_map_to_pairs_float_diagonal_unsafe(backend: str) -> None:
    contact_map = make_contact_map([[1.0, 0.8], [0.8, 0.0]], backend, dtype=float)
    with pytest.warns(UserWarning, match="Diagonal"):
        pairs = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_float_multiple_pairings_unsafe(backend: str) -> None:
    contact_map = make_contact_map([[0.0, 0.9, 0.8], [0.9, 0.0, 0.0], [0.8, 0.0, 0.0]], backend, dtype=float)
    with pytest.warns(UserWarning, match="Multiple pairings"):
        pairs = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_float_upper_only_unsafe(backend: str) -> None:
    contact_map = make_contact_map([[0.0, 0.8], [0.0, 0.0]], backend, dtype=float)
    with pytest.warns(UserWarning, match="not symmetric"):
        pairs = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_float_unsafe(backend: str) -> None:
    out_of_range = make_contact_map([[0.0, 2.0], [-2.0, 0.0]], backend, dtype=float)
    with pytest.warns(UserWarning) as record:
        pairs = notations.contact_map_to_pairs(out_of_range, unsafe=True, threshold=0.1)
    messages = [str(w.message) for w in record]
    assert any("outside [0, 1]" in message for message in messages)
    assert any("not symmetric" in message for message in messages)
    assert as_tuple_list(pairs) == [(0, 1)]


def test_contact_map_to_pairs_float_keep_empty(backend: str) -> None:
    keep_empty = make_contact_map([[0.0, 0.4999999], [0.5000002, 0.0]], backend, dtype=float)
    pairs = notations.contact_map_to_pairs(keep_empty, threshold=0.5)
    assert as_tuple_list(pairs) == []


def test_pairs_to_dot_bracket_roundtrip(backend: str) -> None:
    pairs = make_pairs(CROSSING_PAIRS, backend)
    dot_bracket = notations.pairs_to_dot_bracket(pairs, length=4)
    got = as_set(notations.dot_bracket_to_pairs(dot_bracket))
    assert got == {(0, 2), (1, 3)}


def test_numpy_pairs_to_dot_bracket_empty() -> None:
    dot_bracket = notations.pairs_to_dot_bracket(np.empty((0, 2)), length=3, unsafe=False)
    assert dot_bracket == "..."


def test_numpy_pairs_to_dot_bracket_shape_error() -> None:
    with pytest.raises(ValueError, match="pairs must be"):
        notations.pairs_to_dot_bracket(np.array([1, 2, 3]), length=3, unsafe=False)


def test_numpy_pairs_to_dot_bracket_self_pairing_error() -> None:
    pairs = np.array([[1, 1]])
    with pytest.raises(ValueError, match="Self-pairing"):
        notations.pairs_to_dot_bracket(pairs, length=3, unsafe=False)


def test_numpy_pairs_to_dot_bracket_self_pairing_unsafe() -> None:
    pairs = np.array([[1, 1]])
    with pytest.warns(UserWarning, match="self-pairing"):
        dot_bracket = notations.pairs_to_dot_bracket(pairs, length=3, unsafe=True)
    assert dot_bracket == "..."


def test_numpy_pairs_to_dot_bracket_out_of_bounds_error() -> None:
    oob = np.array([[0, 4]])
    with pytest.raises(ValueError, match="out of bounds"):
        notations.pairs_to_dot_bracket(oob, length=3, unsafe=False)


def test_numpy_pairs_to_dot_bracket_out_of_bounds_unsafe() -> None:
    oob = np.array([[0, 4]])
    with pytest.warns(UserWarning, match="out-of-bounds"):
        dot_bracket = notations.pairs_to_dot_bracket(oob, length=3, unsafe=True)
    assert dot_bracket == "..."


def test_numpy_pairs_to_dot_bracket_duplicates_error() -> None:
    pairs = np.array([[0, 3], [0, 2]])
    with pytest.raises(ValueError, match="paired multiple times"):
        notations.pairs_to_dot_bracket(pairs, length=4, unsafe=False)


def test_numpy_pairs_to_dot_bracket_duplicates_unsafe() -> None:
    pairs = np.array([[0, 3], [0, 2]])
    with pytest.warns(UserWarning, match="multiple pairs"):
        dot_bracket = notations.pairs_to_dot_bracket(pairs, length=4, unsafe=True)
    got = as_set(notations.dot_bracket_to_pairs(dot_bracket))
    assert got == {(0, 2)}


def test_numpy_pairs_to_dot_bracket_too_many_tiers_error() -> None:
    count = len(notations._DOT_BRACKET_PAIR_TABLE) + 1
    pairs = np.array([(idx, idx + count) for idx in range(count)])
    with pytest.raises(ValueError, match="available bracket types"):
        notations.pairs_to_dot_bracket(pairs, length=count * 2, unsafe=False)


def test_numpy_pairs_to_dot_bracket_too_many_tiers_unsafe() -> None:
    count = len(notations._DOT_BRACKET_PAIR_TABLE) + 1
    pairs = np.array([(idx, idx + count) for idx in range(count)])
    with pytest.warns(UserWarning, match="Too many pseudoknot tiers"):
        dot_bracket = notations.pairs_to_dot_bracket(pairs, length=count * 2, unsafe=True)
    expected = {(idx, idx + count) for idx in range(len(notations._DOT_BRACKET_PAIR_TABLE))}
    got = as_set(notations.dot_bracket_to_pairs(dot_bracket))
    assert got == expected


def test_greedy_match_helpers(backend: str) -> None:
    contact_map = make_contact_map(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        backend,
        dtype=bool,
    )
    with pytest.warns(UserWarning, match="paired to multiple"):
        out = notations.contact_map_to_pairs(contact_map, unsafe=True)
    assert as_tuple_list(out) == [(0, 2)]
