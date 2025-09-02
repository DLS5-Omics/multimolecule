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


def _duplex_segments_to_list(segments) -> list[list[tuple[int, int]]]:
    pair_i, pair_j, seg_start, seg_len = segments
    if isinstance(pair_i, (torch.Tensor, np.ndarray)):
        pair_i = pair_i.tolist()
        pair_j = pair_j.tolist()
        seg_start = seg_start.tolist()
        seg_len = seg_len.tolist()
    out = []
    for start, length in zip(seg_start, seg_len):
        start_idx = int(start)
        seg_len = int(length)
        if seg_len <= 0:
            continue
        end_idx = start_idx + seg_len
        out.append([(int(pi), int(pj)) for pi, pj in zip(pair_i[start_idx:end_idx], pair_j[start_idx:end_idx])])
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
def test_pairs_to_duplex_segment_arrays_preserves_bulged_pairs(backend: str) -> None:
    pairs_data = [(2, 20), (3, 17), (4, 16), (5, 15)]
    segments = pairs_utils.pairs_to_duplex_segment_arrays(make_pairs(pairs_data, backend))
    assert as_nested_tuple_lists(_duplex_segments_to_list(segments)) == [[(2, 20), (3, 17), (4, 16), (5, 15)]]


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


@pytest.mark.parametrize("backend", ["list", "numpy", "torch"], ids=["list", "numpy", "torch"])
def test_pairs_to_helix_segment_arrays_basic(backend: str) -> None:
    pairs_data = [(0, 5), (1, 4), (3, 8)]
    pairs = make_pairs(pairs_data, backend)
    start_i, start_j, lengths = pairs_utils.pairs_to_helix_segment_arrays(pairs)
    assert as_list(start_i) == [0, 3]
    assert as_list(start_j) == [5, 8]
    assert as_list(lengths) == [2, 1]


def test_pairs_to_helix_segment_arrays_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        pairs_utils.pairs_to_helix_segment_arrays(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="shape"):
        pairs_utils.pairs_to_helix_segment_arrays([[1, 2, 3]])
    with pytest.raises(TypeError):
        pairs_utils.pairs_to_helix_segment_arrays(object())


@pytest.mark.parametrize("backend", ["list", "numpy", "torch"], ids=["list", "numpy", "torch"])
def test_pairs_to_duplex_segment_arrays_self_pair_empty(backend: str) -> None:
    pairs = make_pairs([(2, 2)], backend)
    pair_i, pair_j, seg_start, seg_len = pairs_utils.pairs_to_duplex_segment_arrays(pairs)
    assert len(as_list(pair_i)) == 0
    assert len(as_list(pair_j)) == 0
    assert len(as_list(seg_start)) == 0
    assert len(as_list(seg_len)) == 0


def test_segment_arrays_to_pairs_duplex_numpy_mask() -> None:
    pairs_data = np.array([[0, 5], [1, 4], [6, 9], [7, 8]], dtype=int)
    segments = pairs_utils.pairs_to_duplex_segment_arrays(pairs_data)
    expected_segments = _duplex_segments_to_list(segments)
    assert len(expected_segments) == 2
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=np.array([1], dtype=int), empty=pairs_data[:0])
    assert as_tuple_list(out) == expected_segments[1]


def test_segment_list_to_pairs_empty_and_sorted() -> None:
    empty = pairs_utils.segment_list_to_pairs([], np.empty((0, 2), dtype=int))
    assert empty.shape == (0, 2)
    empty2 = pairs_utils.segment_list_to_pairs([[]], np.empty((0, 2), dtype=int))
    assert empty2.shape == (0, 2)

    segments = [[(2, 5), (3, 4)], [(0, 7)]]
    out = pairs_utils.segment_list_to_pairs(segments, np.empty((0, 2), dtype=int))
    assert as_tuple_list(out) == [(0, 7), (2, 5), (3, 4)]


def test_stem_segment_arrays_to_stem_segment_list_skips_zero() -> None:
    start_i = np.array([0, 3], dtype=int)
    start_j = np.array([5, 8], dtype=int)
    lengths = np.array([0, 2], dtype=int)
    segments = pairs_utils.stem_segment_arrays_to_stem_segment_list(start_i, start_j, lengths, tier=2)
    assert len(segments) == 1
    assert segments[0].start_5p == 3
    assert segments[0].tier == 2


def test_segment_arrays_to_pairs_tuple_type_errors() -> None:
    with pytest.raises(TypeError):
        pairs_utils.segment_arrays_to_pairs((object(), object(), object()), empty=np.empty((0, 2), dtype=int))
    with pytest.raises(TypeError):
        pairs_utils.segment_arrays_to_pairs((object(), object(), object(), object()), empty=np.empty((0, 2), dtype=int))


def test_pairs_to_stem_segment_arrays_type_error() -> None:
    with pytest.raises(TypeError):
        pairs_utils.pairs_to_stem_segment_arrays(object())


def test_normalize_pairs_list_empty_tuple() -> None:
    assert pairs_utils.normalize_pairs([()]) == []


def test_normalize_pairs_torch_sorted_and_duplicates() -> None:
    pairs = torch.tensor([[0, 3], [1, 2]], dtype=torch.long)
    assert torch.equal(pairs_utils.normalize_pairs(pairs), pairs)

    pairs_dups = torch.tensor([[0, 3], [0, 3], [1, 2]], dtype=torch.long)
    assert as_tuple_list(pairs_utils.normalize_pairs(pairs_dups)) == [(0, 3), (1, 2)]


def test_normalize_pairs_type_error() -> None:
    with pytest.raises(TypeError):
        pairs_utils.normalize_pairs(object())


def test_segment_arrays_to_pairs_torch_mask_indices_and_numpy_bool() -> None:
    pairs = make_pairs([(0, 4), (1, 3), (6, 9), (7, 8)], "torch")
    segments = pairs_utils.pairs_to_stem_segment_arrays(pairs)
    out = pairs_utils.segment_arrays_to_pairs(segments, mask=torch.tensor([1], dtype=torch.long))
    assert as_tuple_list(out) == [(6, 9), (7, 8)]

    out_bool = pairs_utils.segment_arrays_to_pairs(segments, mask=np.array([True, False]))
    assert as_tuple_list(out_bool) == [(0, 4), (1, 3)]


def test_segment_arrays_to_pairs_numpy_mask_variants() -> None:
    pairs_np = np.array([[0, 4], [1, 3], [6, 9], [7, 8]], dtype=int)
    segments = pairs_utils.pairs_to_stem_segment_arrays(pairs_np)
    out_bool = pairs_utils.segment_arrays_to_pairs(segments, mask=[True, False])
    assert as_tuple_list(out_bool) == [(0, 4), (1, 3)]

    out_idx = pairs_utils.segment_arrays_to_pairs(segments, mask=[1])
    assert as_tuple_list(out_idx) == [(6, 9), (7, 8)]


def test_segment_arrays_to_pairs_list_tuple_defaults() -> None:
    out = pairs_utils.segment_arrays_to_pairs(([0], [3], [2]))
    assert as_tuple_list(out) == [(0, 3), (1, 2)]

    pair_i, pair_j, seg_start, seg_len = pairs_utils.pairs_to_duplex_segment_arrays([(0, 5), (1, 4)])
    out_duplex = pairs_utils.segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len))
    assert as_tuple_list(out_duplex) == [(0, 5), (1, 4)]


def test_segment_arrays_to_pairs_all_false_masks() -> None:
    pairs = make_pairs([(0, 4), (1, 3), (6, 9), (7, 8)], "torch")
    start_i, start_j, lengths = pairs_utils.pairs_to_stem_segment_arrays(pairs)
    mask = torch.zeros(int(start_i.numel()), dtype=torch.bool)
    out = pairs_utils.segment_arrays_to_pairs((start_i, start_j, lengths), mask=mask)
    assert out.shape[0] == 0

    pair_i, pair_j, seg_start, seg_len = pairs_utils.pairs_to_duplex_segment_arrays(pairs)
    mask = torch.zeros(int(seg_len.numel()), dtype=torch.bool)
    out = pairs_utils.segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len), mask=mask)
    assert out.shape[0] == 0

    pairs_np = np.array([[0, 4], [1, 3], [6, 9], [7, 8]], dtype=int)
    start_i_np, start_j_np, lengths_np = pairs_utils.pairs_to_stem_segment_arrays(pairs_np)
    mask_np = np.zeros(int(start_i_np.size), dtype=bool)
    out = pairs_utils.segment_arrays_to_pairs((start_i_np, start_j_np, lengths_np), mask=mask_np)
    assert out.shape == (0, 2)

    pair_i_np, pair_j_np, seg_start_np, seg_len_np = pairs_utils.pairs_to_duplex_segment_arrays(pairs_np)
    mask_np = np.zeros(int(seg_len_np.size), dtype=bool)
    out = pairs_utils.segment_arrays_to_pairs((pair_i_np, pair_j_np, seg_start_np, seg_len_np), mask=mask_np)
    assert out.shape == (0, 2)

    segments_list = [[(0, 3)], [(5, 6)]]
    out = pairs_utils.segment_arrays_to_pairs(segments_list, mask=[False, False])
    assert out.shape == (0, 2)


def test_segment_arrays_to_pairs_zero_lengths() -> None:
    start_i = np.array([0], dtype=int)
    start_j = np.array([3], dtype=int)
    lengths = np.array([0], dtype=int)
    out = pairs_utils.segment_arrays_to_pairs((start_i, start_j, lengths))
    assert out.shape == (0, 2)

    pair_i = np.array([0], dtype=int)
    pair_j = np.array([3], dtype=int)
    seg_start = np.array([0], dtype=int)
    seg_len = np.array([0], dtype=int)
    out = pairs_utils.segment_arrays_to_pairs((pair_i, pair_j, seg_start, seg_len))
    assert out.shape == (0, 2)


def test_pairs_helpers_and_duplex_segments() -> None:
    with pytest.raises(ValueError):
        pairs_utils.ensure_pairs_np(torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError):
        pairs_utils.ensure_pairs_np(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        pairs_utils.ensure_pairs_np(np.array([[1, 2, 3]]))
    with pytest.raises(ValueError):
        pairs_utils.ensure_pairs_np(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        pairs_utils.ensure_pairs_np([1, 2, 3])
    with pytest.raises(TypeError):
        pairs_utils.ensure_pairs_np(object())
    assert pairs_utils.ensure_pairs_np(np.empty((0, 2), dtype=int)).shape == (0, 2)
    assert pairs_utils.ensure_pairs_np([[]]).shape == (0, 2)

    assert pairs_utils.ensure_pairs_list(torch.empty((0, 2), dtype=torch.long)) == []
    assert pairs_utils.ensure_pairs_list(np.empty((0, 2), dtype=int)) == []
    assert pairs_utils.ensure_pairs_list([]) == []
    assert pairs_utils.ensure_pairs_list(np.array([[1, 2]], dtype=int)) == [(1, 2)]

    with pytest.raises(ValueError):
        pairs_utils.PairMap([(0, 2)], length=1)
    pair_map = pairs_utils.PairMap([(3, 4)])
    assert pair_map.is_loop_linked(2, 5)
    assert not pair_map.is_loop_linked(0, 5)
    assert len(pair_map.copy().pairs) == len(pair_map.pairs)
    assert pair_map.get(3) == 4
    assert pair_map.get(10, -1) == -1
    assert list(pair_map.keys())
    assert list(pair_map.items())
    assert pair_map.to_list()[0] == -1

    with pytest.raises(ValueError):
        pair_map.to_list(length=1)

    cached = pairs_utils.PairMap([(0, 1)], length=2)
    assert cached.to_list() == [1, 0]
    assert pairs_utils.PairMap([]).to_list() == []

    pair_data = [(0, 5), (1, 4), (3, 7)]
    pair_i, pair_j, seg_start, seg_len = pairs_utils.pairs_to_duplex_segment_arrays(pair_data)
    restored = pairs_utils.segment_arrays_to_pairs(
        (pair_i, pair_j, seg_start, seg_len), empty=np.empty((0, 2), dtype=int)
    )
    assert set(map(tuple, restored.tolist())) == set(pairs_utils.normalize_pairs(pair_data))

    mask_indices = [1]
    subset = pairs_utils.segment_arrays_to_pairs(
        (pair_i, pair_j, seg_start, seg_len), mask=mask_indices, empty=np.empty((0, 2), dtype=int)
    )
    assert subset.shape[0] > 0

    mask_bool = [True] + [False] * (len(seg_len) - 1)
    subset2 = pairs_utils.segment_arrays_to_pairs(
        (pair_i, pair_j, seg_start, seg_len), mask=mask_bool, empty=np.empty((0, 2), dtype=int)
    )
    assert subset2.shape[0] > 0

    torch_pair_i, torch_pair_j, torch_seg_start, torch_seg_len = pairs_utils.pairs_to_duplex_segment_arrays(
        torch.tensor(pair_data)
    )
    empty_pair_i, empty_pair_j, empty_seg_start, empty_seg_len = pairs_utils.pairs_to_duplex_segment_arrays(
        torch.empty((0, 2), dtype=torch.long)
    )
    assert empty_pair_i.numel() == 0
    torch_mask = np.array([0], dtype=int)
    torch_subset = pairs_utils.segment_arrays_to_pairs(
        (torch_pair_i, torch_pair_j, torch_seg_start, torch_seg_len), mask=torch_mask
    )
    assert torch_subset.shape[1] == 2

    with pytest.raises(TypeError):
        pairs_utils.pairs_to_duplex_segment_arrays(object())
    with pytest.raises(ValueError):
        pairs_utils.pairs_to_duplex_segment_arrays(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        pairs_utils.pairs_to_duplex_segment_arrays([[1, 2, 3]])
    with pytest.raises(ValueError):
        pairs_utils.segment_arrays_to_pairs((np.array([0], dtype=int),), empty=np.empty((0, 2), dtype=int))
    with pytest.raises(ValueError):
        pairs_utils.sort_pairs([1, 2, 3])
    with pytest.raises(ValueError):
        pairs_utils.normalize_pairs([1, 2, 3])
