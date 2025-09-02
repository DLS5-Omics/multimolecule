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


import string

import numpy as np
import pytest
import torch

from multimolecule.utils.rna.secondary_structure import notations


def _count_openers(dot_bracket: str) -> set[str]:
    openers = set("([{<" + string.ascii_uppercase)
    return {ch for ch in dot_bracket if ch in openers}


def _pairs_set(pairs) -> set[tuple[int, int]]:
    return set(map(tuple, pairs))


def test_pairs_to_dot_bracket_minimal_tiers() -> None:
    pairs = np.array([(0, 2), (1, 6), (3, 5), (4, 7)], dtype=int)
    dot_bracket = notations.pairs_to_dot_bracket(pairs, length=8)

    assert len(_count_openers(dot_bracket)) == 2
    got = _pairs_set(notations.dot_bracket_to_pairs(dot_bracket).tolist())
    expected = {(0, 2), (1, 6), (3, 5), (4, 7)}
    assert got == expected


def test_dot_bracket_to_pairs_valid_and_errors() -> None:
    dot_bracket = "A+.(a)_"
    pairs = notations.dot_bracket_to_pairs(dot_bracket)
    assert pairs.tolist() == [[0, 4], [3, 5]]

    empty = notations.dot_bracket_to_pairs("...")
    assert empty.shape == (0, 2)

    with pytest.raises(ValueError, match="Unmatched symbol"):
        notations.dot_bracket_to_pairs(")")
    with pytest.raises(ValueError, match="Invalid symbol"):
        notations.dot_bracket_to_pairs("1")
    with pytest.raises(ValueError, match="Unmatched symbol"):
        notations.dot_bracket_to_pairs("(")


def test_dot_bracket_contact_map_roundtrip() -> None:
    dot = "(())"
    contact_map = notations.dot_bracket_to_contact_map(dot)
    assert contact_map.dtype == bool
    assert notations.contact_map_to_dot_bracket(contact_map) == dot


def test_pairs_to_contact_map_numpy_basic() -> None:
    contact_map = notations.pairs_to_contact_map([(2, 0), (1, 3)])
    expected = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(contact_map, expected)


def test_pairs_to_contact_map_numpy_errors_and_warnings() -> None:
    with pytest.raises(ValueError, match="pairs must be"):
        notations.pairs_to_contact_map(np.array([1, 2, 3]))

    pairs = np.array([[1, 1], [0, 2]], dtype=int)
    with pytest.raises(ValueError, match="Self-pairing"):
        notations.pairs_to_contact_map(pairs, length=3)
    with pytest.warns(UserWarning, match="Self-pairing"):
        contact_map = notations.pairs_to_contact_map(pairs, length=3, unsafe=True)
    assert int(contact_map.sum()) == 2

    with pytest.raises(ValueError, match="out of bounds"):
        notations.pairs_to_contact_map(np.array([[0, 3]], dtype=int), length=3)


def test_pairs_to_contact_map_torch_empty_and_errors() -> None:
    empty = torch.empty((0, 2), dtype=torch.long)
    contact_map = notations.pairs_to_contact_map(empty)
    assert contact_map.shape == (0, 0)

    with pytest.raises(ValueError, match="pairs must be"):
        notations.pairs_to_contact_map(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError, match="Self-pairing"):
        notations.pairs_to_contact_map(torch.tensor([[1, 1]], dtype=torch.long), length=2)

    with pytest.warns(UserWarning, match="Self-pairing"):
        contact_map = notations.pairs_to_contact_map(
            torch.tensor([[1, 1], [0, 2]], dtype=torch.long), length=3, unsafe=True
        )
    assert int(contact_map.sum().item()) == 2

    with pytest.raises(ValueError, match="out of bounds"):
        notations.pairs_to_contact_map(torch.tensor([[0, 3]], dtype=torch.long), length=3)


def test_contact_map_to_pairs_binary_torch_paths() -> None:
    with pytest.raises(ValueError, match="square"):
        notations.contact_map_to_pairs(torch.zeros((2, 3), dtype=torch.int64))

    upper_only = torch.tensor([[0, 1], [0, 0]], dtype=torch.int64)
    pairs = notations.contact_map_to_pairs(upper_only)
    assert pairs.tolist() == [[0, 1]]

    diag = torch.tensor([[1, 0], [0, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="diagonal"):
        notations.contact_map_to_pairs(diag)
    with pytest.warns(UserWarning, match="diagonal"):
        pairs = notations.contact_map_to_pairs(diag, unsafe=True)
    assert pairs.numel() == 0

    multi = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="multiple"):
        notations.contact_map_to_pairs(multi)

    not_symmetric = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.int64)
    with pytest.warns(UserWarning) as record:
        pairs = notations.contact_map_to_pairs(not_symmetric, unsafe=True)
    messages = [str(w.message) for w in record]
    assert any("not symmetric" in message for message in messages)
    assert any("paired to multiple" in message for message in messages)
    assert pairs.tolist() == [[0, 1]]

    empty = torch.zeros((0, 0), dtype=torch.int64)
    pairs = notations.contact_map_to_pairs(empty)
    assert pairs.shape == (0, 2)


def test_contact_map_to_pairs_binary_numpy_paths() -> None:
    with pytest.raises(ValueError, match="square"):
        notations.contact_map_to_pairs(np.zeros((2, 3), dtype=int))

    upper_only = np.array([[0, 1], [0, 0]], dtype=int)
    pairs = notations.contact_map_to_pairs(upper_only)
    assert pairs.tolist() == [[0, 1]]

    diag = np.array([[1, 0], [0, 0]], dtype=int)
    with pytest.raises(ValueError, match="diagonal"):
        notations.contact_map_to_pairs(diag)
    with pytest.warns(UserWarning, match="diagonal"):
        pairs = notations.contact_map_to_pairs(diag, unsafe=True)
    assert pairs.size == 0

    multi = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int)
    with pytest.raises(ValueError, match="multiple"):
        notations.contact_map_to_pairs(multi)

    not_symmetric = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=int)
    with pytest.warns(UserWarning) as record:
        pairs = notations.contact_map_to_pairs(not_symmetric, unsafe=True)
    messages = [str(w.message) for w in record]
    assert any("not symmetric" in message for message in messages)
    assert any("paired to multiple" in message for message in messages)
    assert pairs.tolist() == [[0, 1]]

    empty = np.zeros((0, 0), dtype=int)
    pairs = notations.contact_map_to_pairs(empty)
    assert pairs.shape == (0, 2)


def test_contact_map_to_pairs_float_torch_basic_and_empty() -> None:
    contact_map = torch.tensor([[0.0, 0.9], [0.9, 0.0]], dtype=torch.float32)
    pairs = notations.contact_map_to_pairs(contact_map)
    assert pairs.tolist() == [[0, 1]]

    low = torch.tensor([[0.0, 0.4], [0.4, 0.0]], dtype=torch.float32)
    pairs = notations.contact_map_to_pairs(low, threshold=0.5)
    assert pairs.numel() == 0

    empty = torch.empty((0, 0), dtype=torch.float32)
    pairs = notations.contact_map_to_pairs(empty)
    assert pairs.shape == (0, 2)


def test_contact_map_to_pairs_float_torch_errors() -> None:
    with pytest.raises(ValueError, match="square"):
        notations.contact_map_to_pairs(torch.zeros((2, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match="threshold"):
        notations.contact_map_to_pairs(torch.zeros((2, 2), dtype=torch.float32), threshold=1.5)

    with pytest.raises(ValueError, match="NaN"):
        notations.contact_map_to_pairs(torch.tensor([[0.0, float("nan")], [0.0, 0.0]], dtype=torch.float32))


def test_contact_map_to_pairs_float_torch_out_of_range_and_symmetry() -> None:
    out_of_range = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="probabilities"):
        notations.contact_map_to_pairs(out_of_range)

    not_symmetric = torch.tensor([[0.0, 0.9], [0.1, 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="not symmetric"):
        notations.contact_map_to_pairs(not_symmetric)

    upper_only = torch.tensor([[0.0, 0.8], [0.0, 0.0]], dtype=torch.float32)
    pairs = notations.contact_map_to_pairs(upper_only)
    assert pairs.tolist() == [[0, 1]]


def test_contact_map_to_pairs_float_torch_unsafe_and_keep_empty() -> None:
    out_of_range = torch.tensor([[0.0, 2.0], [-2.0, 0.0]], dtype=torch.float32)
    with pytest.warns(UserWarning) as record:
        pairs = notations.contact_map_to_pairs(out_of_range, unsafe=True, threshold=0.1)
    messages = [str(w.message) for w in record]
    assert any("outside [0, 1]" in message for message in messages)
    assert any("not symmetric" in message for message in messages)
    assert pairs.tolist() == [[0, 1]]

    keep_empty = torch.tensor([[0.0, 0.4999999], [0.5000002, 0.0]], dtype=torch.float32)
    pairs = notations.contact_map_to_pairs(keep_empty, threshold=0.5)
    assert pairs.shape == (0, 2)


def test_contact_map_to_pairs_float_numpy_basic_and_empty() -> None:
    contact_map = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float)
    pairs = notations.contact_map_to_pairs(contact_map)
    assert pairs.tolist() == [[0, 1]]

    low = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=float)
    pairs = notations.contact_map_to_pairs(low, threshold=0.5)
    assert pairs.size == 0

    empty = np.empty((0, 0), dtype=float)
    pairs = notations.contact_map_to_pairs(empty)
    assert pairs.shape == (0, 2)


def test_contact_map_to_pairs_float_numpy_errors() -> None:
    with pytest.raises(ValueError, match="square"):
        notations.contact_map_to_pairs(np.zeros((2, 3), dtype=float))

    with pytest.raises(ValueError, match="threshold"):
        notations.contact_map_to_pairs(np.zeros((2, 2), dtype=float), threshold=1.5)

    with pytest.raises(ValueError, match="NaN"):
        notations.contact_map_to_pairs(np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float))


def test_contact_map_to_pairs_float_numpy_out_of_range_and_symmetry() -> None:
    out_of_range = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="probabilities"):
        notations.contact_map_to_pairs(out_of_range)

    not_symmetric = np.array([[0.0, 0.9], [0.1, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="not symmetric"):
        notations.contact_map_to_pairs(not_symmetric)

    upper_only = np.array([[0.0, 0.8], [0.0, 0.0]], dtype=float)
    pairs = notations.contact_map_to_pairs(upper_only)
    assert pairs.tolist() == [[0, 1]]


def test_contact_map_to_pairs_float_numpy_unsafe_and_keep_empty() -> None:
    out_of_range = np.array([[0.0, 2.0], [-2.0, 0.0]], dtype=float)
    with pytest.warns(UserWarning) as record:
        pairs = notations.contact_map_to_pairs(out_of_range, unsafe=True, threshold=0.1)
    messages = [str(w.message) for w in record]
    assert any("outside [0, 1]" in message for message in messages)
    assert any("not symmetric" in message for message in messages)
    assert pairs.tolist() == [[0, 1]]

    keep_empty = np.array([[0.0, 0.4999999], [0.5000002, 0.0]], dtype=float)
    pairs = notations.contact_map_to_pairs(keep_empty, threshold=0.5)
    assert pairs.shape == (0, 2)


def test_pairs_to_dot_bracket_torch_roundtrip() -> None:
    pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    dot_bracket = notations.pairs_to_dot_bracket(pairs, length=4)
    got = _pairs_set(notations.dot_bracket_to_pairs(dot_bracket).tolist())
    assert got == {(0, 2), (1, 3)}


def test_numpy_pairs_to_dot_bracket_empty_and_shape_error() -> None:
    dot_bracket = notations._numpy_pairs_to_dot_bracket(np.empty((0, 2), dtype=int), length=3, unsafe=False)
    assert dot_bracket == "..."

    with pytest.raises(TypeError, match="pairs must be"):
        notations._numpy_pairs_to_dot_bracket(np.array([1, 2, 3]), length=3, unsafe=False)


def test_numpy_pairs_to_dot_bracket_self_pairing_and_bounds() -> None:
    pairs = np.array([[1, 1]], dtype=int)
    with pytest.raises(ValueError, match="Self-pairing"):
        notations._numpy_pairs_to_dot_bracket(pairs, length=3, unsafe=False)
    with pytest.warns(UserWarning, match="self-pairing"):
        dot_bracket = notations._numpy_pairs_to_dot_bracket(pairs, length=3, unsafe=True)
    assert dot_bracket == "..."

    oob = np.array([[0, 4]], dtype=int)
    with pytest.raises(ValueError, match="out of bounds"):
        notations._numpy_pairs_to_dot_bracket(oob, length=3, unsafe=False)
    with pytest.warns(UserWarning, match="out-of-bounds"):
        dot_bracket = notations._numpy_pairs_to_dot_bracket(oob, length=3, unsafe=True)
    assert dot_bracket == "..."


def test_numpy_pairs_to_dot_bracket_duplicates_warning() -> None:
    pairs = np.array([[0, 3], [0, 2]], dtype=int)
    with pytest.raises(ValueError, match="paired multiple times"):
        notations._numpy_pairs_to_dot_bracket(pairs, length=4, unsafe=False)
    with pytest.warns(UserWarning, match="multiple pairs"):
        dot_bracket = notations._numpy_pairs_to_dot_bracket(pairs, length=4, unsafe=True)
    got = _pairs_set(notations.dot_bracket_to_pairs(dot_bracket).tolist())
    assert got == {(0, 2)}


def test_numpy_pairs_to_dot_bracket_too_many_tiers() -> None:
    count = len(notations._DOT_BRACKET_PAIR_TABLE) + 1
    pairs = np.array([(idx, idx + count) for idx in range(count)], dtype=int)
    with pytest.raises(ValueError, match="available bracket types"):
        notations._numpy_pairs_to_dot_bracket(pairs, length=count * 2, unsafe=False)

    with pytest.warns(UserWarning, match="Too many pseudoknot tiers"):
        dot_bracket = notations._numpy_pairs_to_dot_bracket(pairs, length=count * 2, unsafe=True)
    expected = {(idx, idx + count) for idx in range(len(notations._DOT_BRACKET_PAIR_TABLE))}
    got = _pairs_set(notations.dot_bracket_to_pairs(dot_bracket).tolist())
    assert got == expected


def test_crossing_helpers_and_coloring() -> None:
    pairs = np.array([[0, 3], [1, 4], [2, 5]], dtype=int)
    adj = notations._crossing_adjacency(pairs)
    assert adj == [[1, 2], [0, 2], [0, 1]]

    uncolored = [True, True, True]
    neighbor_color_counts = [{0: 1}, {}, {0: 1, 1: 1}]
    degrees = [1, 2, 1]
    assert notations._select_dsatur_vertex(uncolored, neighbor_color_counts, degrees) == 2

    uncolored = [True, True]
    neighbor_color_counts = [{}, {}]
    degrees = [0, 0]
    assert notations._select_dsatur_vertex(uncolored, neighbor_color_counts, degrees) == 0

    colors, num_colors = notations._dsatur_greedy_coloring(adj, [2, 2, 2])
    assert num_colors == 3
    assert sorted(colors) == [0, 1, 2]

    colors_min, num_colors_min = notations._dsatur_min_coloring(adj)
    assert num_colors_min == 3
    assert sorted(colors_min) == [0, 1, 2]

    colors_min, num_colors_min = notations._dsatur_min_coloring([])
    assert colors_min == []
    assert num_colors_min == 0

    colors_min, num_colors_min = notations._dsatur_min_coloring([[], [], []])
    assert num_colors_min == 1
    assert colors_min == [0, 0, 0]

    colors_min, num_colors_min = notations._dsatur_min_coloring([[1], [0]])
    assert num_colors_min == 2
    assert sorted(colors_min) == [0, 1]


def test_pseudoknot_tiers_helpers() -> None:
    pairs = np.array([[0, 3], [1, 4], [2, 5]], dtype=int)
    tiers_greedy = notations._greedy_pseudoknot_tiers(pairs)
    tiers_min = notations._minimal_pseudoknot_tiers(pairs)
    assert len(tiers_greedy) == 3
    assert len(tiers_min) == 3

    assert notations._minimal_pseudoknot_tiers(np.empty((0, 2), dtype=int)) == []
