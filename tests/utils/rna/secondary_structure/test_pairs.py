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


import numpy as np
import pytest
import torch

from multimolecule.utils.rna.secondary_structure import pairs as pairs_utils
from tests.utils.rna.secondary_structure.conftest import (
    as_list,
    as_nested_tuple_lists,
    as_set,
    as_tuple_list,
    make_pairs,
)


def _segments_to_list(segments) -> list[list[tuple[int, int]]]:
    seg_i, seg_j, seg_len = segments
    out = []
    if isinstance(seg_i, torch.Tensor):
        count = int(seg_i.numel())
    elif isinstance(seg_i, np.ndarray):
        count = int(seg_i.size)
    else:
        count = len(seg_i)
    for idx in range(count):
        length = int(seg_len[idx])
        if length <= 0:
            continue
        start_i = int(seg_i[idx])
        start_j = int(seg_j[idx])
        out.append([(start_i + offset, start_j - offset) for offset in range(length)])
    return out


@pytest.mark.parametrize(
    "pairs, expected",
    [
        pytest.param([[3, 1], [1, 3], [2, 0]], [(0, 2), (1, 3)], id="unsorted_with_swaps"),
        pytest.param([[0, 2], [0, 2], [1, 3]], [(0, 2), (1, 3)], id="duplicate_pairs"),
        pytest.param([[0, 2], [1, 3]], [(0, 2), (1, 3)], id="already_sorted"),
    ],
)
def test_normalize_pairs_consistency(backend: str, pairs, expected) -> None:
    normalized = pairs_utils.normalize_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(normalized) == expected


def test_normalize_pairs_empty(backend: str) -> None:
    normalized = pairs_utils.normalize_pairs(make_pairs([], backend))
    if backend == "list":
        assert normalized == []
    else:
        assert normalized.shape == (0, 2)


@pytest.mark.parametrize(
    "bad_pairs",
    [np.array([1, 2, 3]), torch.tensor([1, 2, 3])],
    ids=["numpy_1d", "torch_1d"],
)
def test_normalize_pairs_shape_errors(bad_pairs) -> None:
    with pytest.raises(ValueError, match="shape"):
        pairs_utils.normalize_pairs(bad_pairs)


def test_sort_pairs_consistency(backend: str) -> None:
    pairs = make_pairs([[2, 5], [0, 1], [2, 3]], backend)
    sorted_pairs = pairs_utils.sort_pairs(pairs)
    assert as_tuple_list(sorted_pairs) == [(0, 1), (2, 3), (2, 5)]


def test_sort_pairs_empty(backend: str) -> None:
    sorted_pairs = pairs_utils.sort_pairs(make_pairs([], backend))
    if backend == "list":
        assert sorted_pairs == []
    else:
        assert sorted_pairs.shape == (0, 2)


@pytest.mark.parametrize(
    "bad_pairs",
    [np.array([1, 2, 3]), torch.tensor([1, 2, 3])],
    ids=["numpy_1d", "torch_1d"],
)
def test_sort_pairs_shape_errors(bad_pairs) -> None:
    with pytest.raises(ValueError, match="shape"):
        pairs_utils.sort_pairs(bad_pairs)


@pytest.mark.parametrize("backend", ["list", "numpy"], ids=["list", "numpy"])
def test_pairs_segments_roundtrip_numpy_list(backend: str) -> None:
    pairs_data = [(0, 4), (1, 3), (6, 9), (7, 8)]
    pairs = make_pairs(pairs_data, backend)
    segments = pairs_utils.pairs_to_stem_segment_arrays(pairs)
    rebuilt = pairs_utils.segment_arrays_to_pairs(segments, empty=np.empty((0, 2), dtype=int))
    assert as_set(rebuilt) == set(pairs_data)
    roundtrip_segments = pairs_utils.pairs_to_stem_segment_arrays(rebuilt)
    assert as_nested_tuple_lists(_segments_to_list(roundtrip_segments)) == as_nested_tuple_lists(
        _segments_to_list(segments)
    )


def test_pairs_segments_roundtrip_torch() -> None:
    pairs_data = [(0, 4), (1, 3), (6, 9), (7, 8)]
    pairs = make_pairs(pairs_data, "torch")
    segments = pairs_utils.pairs_to_stem_segment_arrays(pairs)
    rebuilt = pairs_utils.segment_arrays_to_pairs(segments)
    assert as_tuple_list(rebuilt) == pairs_data
    seg_i, seg_j, seg_len = segments
    seg_i2, seg_j2, seg_len2 = pairs_utils.pairs_to_stem_segment_arrays(rebuilt)
    assert as_list(seg_i2) == as_list(seg_i)
    assert as_list(seg_j2) == as_list(seg_j)
    assert as_list(seg_len2) == as_list(seg_len)


def test_pairs_segments_gap_consistency() -> None:
    pairs_data = [(0, 7), (1, 6), (3, 4)]
    segments_list = pairs_utils.pairs_to_stem_segment_arrays(pairs_data)
    segments_np = pairs_utils.pairs_to_stem_segment_arrays(np.array(pairs_data))
    segments_torch = pairs_utils.pairs_to_stem_segment_arrays(torch.tensor(pairs_data))
    expected = [[(0, 7), (1, 6), (2, 5)]]
    assert as_nested_tuple_lists(_segments_to_list(segments_list)) == expected
    assert as_nested_tuple_lists(_segments_to_list(segments_np)) == expected
    assert as_nested_tuple_lists(_segments_to_list(segments_torch)) == expected


@pytest.mark.parametrize("backend", ["list", "numpy", "torch"], ids=["list", "numpy", "torch"])
def test_pairs_to_duplex_segment_list_preserves_bulged_pairs(backend: str) -> None:
    pairs_data = [(2, 20), (3, 17), (4, 16), (5, 15)]
    segments = pairs_utils.pairs_to_duplex_segment_list(make_pairs(pairs_data, backend))
    assert as_nested_tuple_lists(segments) == [[(2, 20), (3, 17), (4, 16), (5, 15)]]


def test_pairs_segments_empty(backend: str) -> None:
    pairs = make_pairs([], backend)
    segments = pairs_utils.pairs_to_stem_segment_arrays(pairs)
    seg_i, seg_j, seg_len = segments
    assert as_list(seg_i) == []
    assert as_list(seg_j) == []
    assert as_list(seg_len) == []


@pytest.mark.parametrize(
    "pairs",
    [np.array([1, 2, 3]), np.array([[0, 1, 2]]), [1, 2, 3], [(0, 1, 2)]],
    ids=["numpy_1d", "numpy_2d_bad", "list_1d", "list_2d_bad"],
)
def test_pairs_to_stem_segment_arrays_shape_errors(pairs) -> None:
    with pytest.raises(ValueError, match="shape"):
        pairs_utils.pairs_to_stem_segment_arrays(pairs)


def test_segment_arrays_to_pairs_mask_indices() -> None:
    segments = [[(0, 3), (1, 2)], [(4, 6)]]
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=[1], empty=np.empty((0, 2), dtype=int))
    assert as_tuple_list(out) == [(4, 6)]


def test_segment_arrays_to_pairs_mask_bool_list() -> None:
    segments = [[(0, 3), (1, 2)], [(4, 6)]]
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=[True, False], empty=np.empty((0, 2), dtype=int))
    assert as_tuple_list(out) == [(0, 3), (1, 2)]


def test_segment_arrays_to_pairs_mask_numpy_bool() -> None:
    segments = [[(0, 3), (1, 2)], [(4, 6)]]
    mask = np.array([True, False])
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=mask, empty=np.empty((0, 2), dtype=int))
    assert as_tuple_list(out) == [(0, 3), (1, 2)]


def test_segment_arrays_to_pairs_mask_numpy_indices() -> None:
    segments = [[(0, 3), (1, 2)], [(4, 6)]]
    mask = np.array([1], dtype=int)
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=mask, empty=np.empty((0, 2), dtype=int))
    assert as_tuple_list(out) == [(4, 6)]


def test_segment_arrays_to_pairs_mask_type_error() -> None:
    segments = [[(0, 1)]]
    with pytest.raises(TypeError, match="mask must be"):
        pairs_utils.segment_arrays_to_pairs(segments, mask="bad", empty=np.empty((0, 2), dtype=int))


def test_segment_arrays_to_pairs_torch_mask_filters() -> None:
    pairs = make_pairs([(0, 4), (1, 3), (6, 9), (7, 8)], "torch")
    segments = pairs_utils.pairs_to_stem_segment_arrays(pairs)
    mask = torch.tensor([True, False])
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=mask)
    assert as_tuple_list(out) == [(0, 4), (1, 3)]
