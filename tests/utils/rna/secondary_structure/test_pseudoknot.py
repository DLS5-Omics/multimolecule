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

from multimolecule.utils.rna.secondary_structure import pseudoknot
from tests.utils.rna.secondary_structure.conftest import (
    CROSSING_PAIRS,
    as_list,
    as_nested_tuple_lists,
    as_set,
    as_tuple_list,
    make_pairs,
)

NESTED_PAIRS = [(0, 3), (1, 2)]


@pytest.mark.parametrize(
    "pairs, expected_primary, expected_pseudoknot",
    [
        pytest.param(CROSSING_PAIRS, [(1, 3)], [(0, 2)], id="crossing"),
        pytest.param(NESTED_PAIRS, NESTED_PAIRS, [], id="nested"),
        pytest.param([(0, 3)], [(0, 3)], [], id="single_pair"),
        pytest.param([], [], [], id="empty"),
    ],
)
def test_split_pseudoknot_pairs_consistency(
    backend: str,
    pairs,
    expected_primary,
    expected_pseudoknot,
) -> None:
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == expected_primary
    assert as_tuple_list(pseudoknot_pairs) == expected_pseudoknot


def test_pseudoknot_pairs_helpers(backend: str) -> None:
    pairs = make_pairs(CROSSING_PAIRS, backend)
    assert as_tuple_list(pseudoknot.nested_pairs(pairs)) == [(1, 3)]
    assert as_tuple_list(pseudoknot.pseudoknot_pairs(pairs)) == [(0, 2)]
    assert as_tuple_list(pseudoknot.crossing_pairs(pairs)) == [(0, 2), (1, 3)]


def test_split_pseudoknot_pairs_input_type_error() -> None:
    with pytest.raises(TypeError):
        pseudoknot.split_pseudoknot_pairs(123)


def test_pseudoknot_pairs_tie_break_unpaired_within_span(backend: str) -> None:
    pairs = [(1, 57), (28, 169)]  # AWSQ01000004.1_575187-575435
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == [(1, 57)]
    assert as_tuple_list(pseudoknot_pairs) == [(28, 169)]


def test_pseudoknot_pairs_tie_break_span(backend: str) -> None:
    pairs = [(1, 8), (2, 7), (0, 3), (6, 9)]  # Constructed span tie.
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == [(0, 3), (6, 9)]
    assert as_tuple_list(pseudoknot_pairs) == [(1, 8), (2, 7)]


def test_pseudoknot_pairs_tie_break_deterministic(backend: str) -> None:
    pairs = [(0, 57), (21, 78)]  # JTFG03009699.1_10109-9752
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == [(21, 78)]
    assert as_tuple_list(pseudoknot_pairs) == [(0, 57)]


def test_pseudoknot_tiers_consistency(backend: str) -> None:
    pairs = make_pairs(CROSSING_PAIRS, backend)
    tiers = pseudoknot.pseudoknot_tiers(pairs)
    flat = [tuple(pair) for tier in as_nested_tuple_lists(tiers) for pair in tier]
    assert set(flat) == {(0, 2), (1, 3)}
    assert len(tiers) == 2

    tiers_unsafe = pseudoknot.pseudoknot_tiers(pairs, unsafe=True)
    assert as_nested_tuple_lists(tiers) == as_nested_tuple_lists(tiers_unsafe)


def test_pseudoknot_tiers_empty(backend: str) -> None:
    tiers = pseudoknot.pseudoknot_tiers(make_pairs([], backend))
    assert tiers == []


def test_crossing_helpers_and_coloring() -> None:
    pairs = np.array([[0, 3], [1, 4], [2, 5]])
    tiers_min = pseudoknot.pseudoknot_tiers(pairs)
    tiers_unsafe = pseudoknot.pseudoknot_tiers(pairs, unsafe=True)
    assert len(tiers_min) == 3
    assert len(tiers_unsafe) == 3
    flat = [pair for tier in as_nested_tuple_lists(tiers_min) for pair in tier]
    assert set(flat) == set(as_tuple_list(pairs))


def test_pseudoknot_tiers_helpers() -> None:
    pairs = np.array([[0, 3], [1, 4], [2, 5]])
    tiers_greedy = pseudoknot.pseudoknot_tiers(pairs, unsafe=True)
    tiers_min = pseudoknot.pseudoknot_tiers(pairs, unsafe=False)
    assert len(tiers_greedy) == 3
    assert len(tiers_min) == 3

    assert pseudoknot.pseudoknot_tiers(np.empty((0, 2))) == []


def test_pseudoknot_nucleotides_and_crossing_nucleotides_consistency(backend: str) -> None:
    pairs = make_pairs(CROSSING_PAIRS, backend)
    assert as_list(pseudoknot.pseudoknot_nucleotides(pairs)) == [0, 2]
    assert as_list(pseudoknot.crossing_nucleotides(pairs)) == [0, 1, 2, 3]

    empty = make_pairs([], backend)
    assert as_list(pseudoknot.pseudoknot_nucleotides(empty)) == []

    non_crossing = make_pairs([(0, 3), (1, 2)], backend)
    assert as_list(pseudoknot.crossing_nucleotides(non_crossing)) == []


def test_mwis_select_basic(backend: str) -> None:
    pairs = [(0, 5), (1, 4), (2, 7)]
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == [(0, 5), (1, 4)]
    assert as_tuple_list(pseudoknot_pairs) == [(2, 7)]


def test_split_pseudoknot_pairs_multi_segment_consistency(backend: str) -> None:
    pairs = [(0, 4), (1, 3), (5, 8), (6, 7)]
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == [(0, 4), (1, 3), (5, 8), (6, 7)]
    assert as_tuple_list(pseudoknot_pairs) == []


def test_split_pseudoknot_pairs_preserves_input_pairs(backend: str) -> None:
    pairs = [(2, 20), (3, 17), (4, 16), (5, 15), (8, 24), (9, 23)]
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(make_pairs(pairs, backend))
    assert as_set(nested_pairs) | as_set(pseudoknot_pairs) == set(pairs)


def test_split_pseudoknot_pairs_expected_segments(backend: str) -> None:
    pairs = make_pairs([(0, 6), (1, 5), (2, 8), (3, 7)], backend)
    assert as_tuple_list(pseudoknot.crossing_pairs(pairs)) == [(0, 6), (1, 5), (2, 8), (3, 7)]
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(pairs)
    assert as_tuple_list(nested_pairs) == [(2, 8), (3, 7)]
    assert as_tuple_list(pseudoknot_pairs) == [(0, 6), (1, 5)]


def test_split_pseudoknot_pairs_self_pairs(backend: str) -> None:
    self_pairs = make_pairs([(1, 1), (2, 2)], backend)
    nested_pairs, pseudoknot_pairs = pseudoknot.split_pseudoknot_pairs(self_pairs)
    assert as_tuple_list(nested_pairs) == [(1, 1), (2, 2)]
    assert as_tuple_list(pseudoknot_pairs) == []


def test_crossing_pairs_segment_path(backend: str) -> None:
    pairs = make_pairs([(0, 6), (1, 5), (2, 8), (3, 7)], backend)
    assert as_tuple_list(pseudoknot.crossing_pairs(pairs)) == [(0, 6), (1, 5), (2, 8), (3, 7)]


def test_crossing_pairs_non_crossing_empty(backend: str) -> None:
    pairs = make_pairs([(0, 4), (1, 3)], backend)
    assert as_tuple_list(pseudoknot.crossing_pairs(pairs)) == []


@pytest.mark.parametrize(
    "pairs, expected_primary, expected_crossing",
    [
        pytest.param(CROSSING_PAIRS, [], CROSSING_PAIRS, id="crossing"),
        pytest.param(NESTED_PAIRS, NESTED_PAIRS, [], id="nested"),
    ],
)
def test_split_crossing_pairs_basic(
    backend: str,
    pairs,
    expected_primary,
    expected_crossing,
) -> None:
    nested_pairs, crossing_pairs = pseudoknot.split_crossing_pairs(make_pairs(pairs, backend))
    assert as_tuple_list(nested_pairs) == expected_primary
    assert as_tuple_list(crossing_pairs) == expected_crossing


def test_crossing_mask_small_consistency(backend: str) -> None:
    pairs = make_pairs([(0, 4), (1, 5), (2, 3)], backend)
    mask = pseudoknot.crossing_mask(pairs)
    assert as_list(mask) == [True, True, False]


def test_crossing_mask_unsorted_pairs() -> None:
    unsorted_pairs = np.array([[2, 5], [0, 3], [1, 4]])
    mask = pseudoknot.crossing_mask(unsorted_pairs)
    assert as_list(mask) == [True, True, True]


def test_crossing_mask_unsorted_pairs_torch() -> None:
    unsorted_pairs = torch.tensor([[5, 6], [1, 4], [0, 3]])
    mask = pseudoknot.crossing_mask(unsorted_pairs)
    assert as_list(mask) == [False, True, True]


def test_crossing_mask_single_and_empty(backend: str) -> None:
    single = make_pairs([(0, 2)], backend)
    assert as_list(pseudoknot.crossing_mask(single)) == [False]

    empty = make_pairs([], backend)
    assert as_list(pseudoknot.crossing_mask(empty)) == []


def test_crossing_pairs_empty(backend: str) -> None:
    empty_pairs = make_pairs([], backend)
    out = pseudoknot.crossing_pairs(empty_pairs)
    assert as_tuple_list(out) == []


def test_pseudoknot_input_type_errors() -> None:
    funcs = (
        pseudoknot.crossing_pairs,
        pseudoknot.crossing_nucleotides,
        pseudoknot.pseudoknot_tiers,
        pseudoknot.pseudoknot_nucleotides,
    )
    for func in funcs:
        with pytest.raises(TypeError):
            func(123)


@pytest.mark.parametrize(
    "func",
    [
        pseudoknot.split_pseudoknot_pairs,
        pseudoknot.split_crossing_pairs,
        pseudoknot.nested_pairs,
        pseudoknot.pseudoknot_pairs,
        pseudoknot.crossing_pairs,
        pseudoknot.crossing_events,
        pseudoknot.crossing_arcs,
        pseudoknot.crossing_mask,
    ],
)
def test_pseudoknot_shape_errors_numpy(func) -> None:
    with pytest.raises(ValueError):
        func(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        func(np.array([[1, 2, 3]]))


@pytest.mark.parametrize(
    "func",
    [
        pseudoknot.split_pseudoknot_pairs,
        pseudoknot.split_crossing_pairs,
        pseudoknot.nested_pairs,
        pseudoknot.pseudoknot_pairs,
        pseudoknot.crossing_pairs,
        pseudoknot.crossing_events,
        pseudoknot.crossing_arcs,
        pseudoknot.crossing_mask,
    ],
)
def test_pseudoknot_shape_errors_list(func) -> None:
    with pytest.raises(ValueError):
        func([1, 2, 3])


@pytest.mark.parametrize(
    "func",
    [
        pseudoknot.split_pseudoknot_pairs,
        pseudoknot.split_crossing_pairs,
        pseudoknot.nested_pairs,
        pseudoknot.pseudoknot_pairs,
        pseudoknot.crossing_pairs,
        pseudoknot.crossing_events,
        pseudoknot.crossing_arcs,
        pseudoknot.crossing_mask,
    ],
)
def test_pseudoknot_shape_errors_torch(func) -> None:
    with pytest.raises(ValueError):
        func(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        func(torch.tensor([[1, 2, 3]]))


@pytest.mark.parametrize("backend", ["list", "numpy", "torch"], ids=["list", "numpy", "torch"])
def test_crossing_events_and_arcs(backend: str) -> None:
    pairs = make_pairs(CROSSING_PAIRS, backend)

    events = pseudoknot.crossing_events(pairs)
    if isinstance(events, list):
        normalized_events = [tuple(map(tuple, event)) for event in events]
    else:
        normalized_events = [tuple(map(tuple, event)) for event in as_list(events)]
    assert normalized_events == [((0, 0, 2, 2), (1, 1, 3, 3))]

    arcs = pseudoknot.crossing_arcs(pairs)
    if isinstance(arcs, list):
        normalized_arcs = [tuple(map(tuple, arc)) for arc in arcs]
    else:
        normalized_arcs = [tuple(map(tuple, arc)) for arc in as_list(arcs)]
    assert normalized_arcs == [((0, 2), (1, 3))]


def test_crossing_events_empty_and_non_crossing() -> None:
    assert pseudoknot.crossing_events([]) == []

    nested = torch.tensor(NESTED_PAIRS, dtype=torch.long)
    events = pseudoknot.crossing_events(nested)
    assert events.numel() == 0

    nested_np = np.array(NESTED_PAIRS, dtype=int)
    assert pseudoknot.crossing_events(nested_np).size == 0


def test_crossing_arcs_empty_list() -> None:
    assert pseudoknot.crossing_arcs([]) == []


def test_has_pseudoknot() -> None:
    assert pseudoknot.has_pseudoknot(CROSSING_PAIRS)
    assert not pseudoknot.has_pseudoknot(NESTED_PAIRS)
    assert not pseudoknot.has_pseudoknot([])
    assert not pseudoknot.has_pseudoknot([(0, 3)])
    assert not pseudoknot.has_pseudoknot([(0, 3), (1, 2)])


def test_crossing_pairs_single_and_duplicate() -> None:
    assert as_tuple_list(pseudoknot.crossing_pairs([(0, 3)])) == []
    assert as_tuple_list(pseudoknot.crossing_pairs([(0, 3), (0, 3)])) == []


def test_crossing_mask_large_paths() -> None:
    count = pseudoknot._CROSSING_N2_THRESHOLD + 1
    base_pairs = [(0, 3), (1, 4)] + [(idx, idx + 1) for idx in range(2, count)]
    pairs_np = np.array(base_pairs)
    mask_np = pseudoknot.crossing_mask(pairs_np)
    pairs_pt = torch.tensor(base_pairs)
    mask_pt = pseudoknot.crossing_mask(pairs_pt)
    assert as_list(mask_pt) == as_list(mask_np)


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([], []),
        ([(0, 0)], [False]),
        ([(0, 1), (2, 3)], [False, False]),
    ],
    ids=["empty", "self_pair", "adjacent"],
)
def test_crossing_mask_edge_cases(pairs, expected) -> None:
    if pairs:
        tensor = torch.tensor(pairs, dtype=torch.long)
    else:
        tensor = torch.empty((0, 2), dtype=torch.long)
    assert as_list(pseudoknot.crossing_mask(tensor)) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_torch_crossing_mask_cuda_small() -> None:
    pairs = torch.tensor([[0, 2], [1, 3]], device="cuda")
    mask = pseudoknot.crossing_mask(pairs)
    assert as_list(mask) == [True, True]
