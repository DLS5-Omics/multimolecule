# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
#
# This file is part of MultiMolecule.
#
# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

import numpy as np
import pytest
import torch

from multimolecule.utils.rna.secondary_structure import noncanonical
from tests.utils.rna.secondary_structure.conftest import as_list, as_tuple_list, make_indices, make_pairs

SEQUENCE_ACGU = "ACGU"
SEQUENCE_ACGA = "ACGA"
SEQUENCE_ACGN = "ACGN"


@pytest.mark.parametrize(
    "sequence, expected",
    [
        pytest.param("", [], id="empty"),
        pytest.param("aCgU", [0, 1, 2, 3], id="case_insensitive"),
        pytest.param("N", [-1], id="unknown"),
    ],
)
def test_sequence_codes(sequence: str, expected: list[int]) -> None:
    codes = noncanonical._sequence_codes(sequence)
    assert as_list(codes) == expected


def test_canonical_table_torch_cache_reuse() -> None:
    original_cache = dict(noncanonical._CANONICAL_TABLE_TORCH_CACHE)
    try:
        noncanonical._CANONICAL_TABLE_TORCH_CACHE.clear()
        device = torch.device("cpu")
        table1 = noncanonical._canonical_table_torch(device)
        table2 = noncanonical._canonical_table_torch(device)
        assert table1.device.type == "cpu"
        assert table1.dtype == torch.bool
        assert table1 is table2
    finally:
        noncanonical._CANONICAL_TABLE_TORCH_CACHE.clear()
        noncanonical._CANONICAL_TABLE_TORCH_CACHE.update(original_cache)


def test_noncanonical_pairs_filters_noncanonical(backend: str) -> None:
    pairs = make_pairs([[0, 3], [0, 1], [2, 3]], backend)
    assert as_tuple_list(noncanonical.noncanonical_pairs(pairs, SEQUENCE_ACGU)) == [(0, 1)]


def test_noncanonical_pairs_filters_canonical(backend: str) -> None:
    canonical_pairs = make_pairs([[0, 3], [1, 2]], backend)
    assert as_tuple_list(noncanonical.noncanonical_pairs(canonical_pairs, SEQUENCE_ACGU)) == []


def test_noncanonical_pairs_unknown(backend: str) -> None:
    pairs = make_pairs([[2, 3]], backend)
    assert as_tuple_list(noncanonical.noncanonical_pairs(pairs, SEQUENCE_ACGN)) == [(2, 3)]


def test_noncanonical_pairs_empty(backend: str) -> None:
    empty_pairs = make_pairs([], backend)
    assert as_tuple_list(noncanonical.noncanonical_pairs(empty_pairs, SEQUENCE_ACGU)) == []


def test_noncanonical_pairs_set_variants(backend: str) -> None:
    pairs = make_pairs([(0, 3), (1, 2)], backend)
    assert noncanonical.noncanonical_pairs_set(pairs, SEQUENCE_ACGA) == {(0, 3)}


def test_noncanonical_pairs_set_empty(backend: str) -> None:
    assert noncanonical.noncanonical_pairs_set(make_pairs([], backend), SEQUENCE_ACGA) == set()


def test_noncanonical_pairs_set_unsafe_self_pair() -> None:
    assert noncanonical.noncanonical_pairs_set([(1, 1)], SEQUENCE_ACGA, unsafe=True) == set()


def test_noncanonical_pairs_set_shape_error() -> None:
    with pytest.raises(ValueError, match="index tuples"):
        noncanonical.noncanonical_pairs_set(np.array([1, 2, 3]), SEQUENCE_ACGA)


def test_noncanonical_invalid_types_and_masks() -> None:
    with pytest.raises(TypeError):
        noncanonical.noncanonical_pairs(object(), SEQUENCE_ACGU)
    with pytest.raises(TypeError):
        noncanonical.noncanonical_pairs_set(object(), SEQUENCE_ACGU)
    assert noncanonical.noncanonical_pairs_set([], SEQUENCE_ACGU) == set()
    assert noncanonical.noncanonical_pairs_set(np.empty((0, 2), dtype=int), SEQUENCE_ACGU) == set()
    assert noncanonical.noncanonical_pairs_set([[]], SEQUENCE_ACGU) == set()
    with pytest.raises(ValueError):
        noncanonical.noncanonical_pair_mask([0, 1], [0], SEQUENCE_ACGU)


@pytest.mark.parametrize(
    "pairs",
    [[[1, 2, 3]], np.array([1, 2, 3]), torch.tensor([1, 2, 3])],
    ids=["list", "numpy", "torch"],
)
def test_noncanonical_pairs_errors(pairs) -> None:
    with pytest.raises(ValueError, match="shape"):
        noncanonical.noncanonical_pairs(pairs, SEQUENCE_ACGU)


def test_noncanonical_pair_mask_canonical(backend: str) -> None:
    pair_i = make_indices([0, 1], backend)
    pair_j = make_indices([3, 2], backend)
    mask = noncanonical.noncanonical_pair_mask(pair_i, pair_j, SEQUENCE_ACGU)
    assert as_list(mask) == [False, False]


def test_noncanonical_pair_mask_unknown(backend: str) -> None:
    pair_i = make_indices([2], backend)
    pair_j = make_indices([3], backend)
    mask = noncanonical.noncanonical_pair_mask(pair_i, pair_j, SEQUENCE_ACGN)
    assert as_list(mask) == [True]


def test_noncanonical_pair_mask_empty_sequence(backend: str) -> None:
    pair_i = make_indices([0, 1], backend)
    pair_j = make_indices([1, 2], backend)
    mask = noncanonical.noncanonical_pair_mask(pair_i, pair_j, "")
    assert as_list(mask) == [False, False]


def test_noncanonical_pair_mask_empty_pairs(backend: str) -> None:
    empty = make_indices([], backend)
    assert as_list(noncanonical.noncanonical_pair_mask(empty, empty, SEQUENCE_ACGU)) == []


def test_noncanonical_pair_mask_self_pairs_error(backend: str) -> None:
    pair_i = make_indices([1], backend)
    pair_j = make_indices([1], backend)
    with pytest.raises(ValueError, match="Self-pairing"):
        noncanonical.noncanonical_pair_mask(pair_i, pair_j, SEQUENCE_ACGU)


def test_noncanonical_pair_mask_self_pairs_unsafe(backend: str) -> None:
    pair_i = make_indices([1], backend)
    pair_j = make_indices([1], backend)
    with pytest.warns(UserWarning, match="self-pairing"):
        mask = noncanonical.noncanonical_pair_mask(pair_i, pair_j, SEQUENCE_ACGU, unsafe=True)
    assert as_list(mask) == [False]


@pytest.mark.parametrize(
    "pair_i, pair_j, exc_type, match",
    [
        pytest.param(
            np.array([0, 1]),
            np.array([2]),
            ValueError,
            "same shape",
            id="numpy_shape_mismatch",
        ),
        pytest.param(
            torch.tensor([0, 1]),
            torch.tensor([2]),
            ValueError,
            "same shape",
            id="torch_shape_mismatch",
        ),
        pytest.param(
            torch.tensor([0]),
            np.array([1]),
            TypeError,
            "same type",
            id="type_mismatch",
        ),
    ],
)
def test_noncanonical_pair_mask_shape_and_type_errors(pair_i, pair_j, exc_type, match) -> None:
    with pytest.raises(exc_type, match=match):
        noncanonical.noncanonical_pair_mask(pair_i, pair_j, SEQUENCE_ACGU)


def test_noncanonical_pair_mask_device_mismatch() -> None:
    pair_i = torch.tensor([0], device="cpu")
    pair_j = torch.empty((1,), device="meta")
    with pytest.raises(ValueError, match="same device"):
        noncanonical.noncanonical_pair_mask(pair_i, pair_j, SEQUENCE_ACGU)
