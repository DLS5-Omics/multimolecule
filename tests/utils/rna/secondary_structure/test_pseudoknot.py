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


def test_primary_and_pseudoknot_pairs_list_outputs() -> None:
    primary_pairs, pk_pairs = pseudoknot.split_pseudoknot_pairs([[0, 3], [1, 2]])
    assert isinstance(primary_pairs, list)
    assert isinstance(pk_pairs, list)
    assert primary_pairs == [[0, 3], [1, 2]]
    assert pk_pairs == []

    primary_pairs, pk_pairs = pseudoknot.split_pseudoknot_pairs([[0, 2], [1, 3]])
    assert primary_pairs == [[0, 2]]
    assert pk_pairs == [[1, 3]]

    assert pseudoknot.primary_pairs([[0, 2], [1, 3]]) == [[0, 2]]
    assert pseudoknot.pseudoknot_pairs([[0, 2], [1, 3]]) == [[1, 3]]
    assert pseudoknot.crossing_pairs([[0, 2], [1, 3]]) == [[0, 2], [1, 3]]


def test_primary_and_pseudoknot_pairs_numpy_and_torch() -> None:
    pairs_np = np.array([[0, 2], [1, 3]], dtype=int)
    primary_np, pk_np = pseudoknot.split_pseudoknot_pairs(pairs_np)
    assert primary_np.tolist() == [[0, 2]]
    assert pk_np.tolist() == [[1, 3]]

    primary_numpy_only = pseudoknot.primary_pairs(np.array([[0, 3], [1, 2]], dtype=int))
    pk_numpy_only = pseudoknot.pseudoknot_pairs(np.array([[0, 3], [1, 2]], dtype=int))
    assert primary_numpy_only.shape == (2, 2)
    assert pk_numpy_only.size == 0

    pairs_torch = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    primary_torch, pk_torch = pseudoknot.split_pseudoknot_pairs(pairs_torch)
    assert primary_torch.tolist() == [[0, 2]]
    assert pk_torch.tolist() == [[1, 3]]

    primary_torch_only = pseudoknot.primary_pairs(torch.tensor([[0, 2], [1, 3]], dtype=torch.long))
    pk_torch_only = pseudoknot.pseudoknot_pairs(torch.tensor([[0, 2], [1, 3]], dtype=torch.long))
    assert primary_torch_only.tolist() == [[0, 2]]
    assert pk_torch_only.tolist() == [[1, 3]]


def test_pseudoknot_tiers_public() -> None:
    tiers = pseudoknot.pseudoknot_tiers(np.array([[0, 2], [1, 3]], dtype=int))
    assert len(tiers) == 2
    assert tiers[0].tolist() == [[0, 2]]
    assert tiers[1].tolist() == [[1, 3]]

    tiers_list = pseudoknot.pseudoknot_tiers([[0, 3], [1, 2]])
    assert len(tiers_list) == 1
    assert tiers_list[0] == [[0, 3], [1, 2]]


def test_pseudoknot_tiers_unsafe_list() -> None:
    tiers = pseudoknot.pseudoknot_tiers([[0, 2], [1, 3]], unsafe=True)
    assert len(tiers) == 2
    flat = sorted(tuple(pair) for tier in tiers for pair in tier)
    assert flat == [(0, 2), (1, 3)]


def test_pseudoknot_nucleotides_numpy_and_torch() -> None:
    pairs_np = np.array([[0, 2], [1, 3]], dtype=int)
    pk_np = pseudoknot.pseudoknot_nucleotides(pairs_np)
    assert pk_np.tolist() == [1, 3]

    pairs_torch = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    pk_torch = pseudoknot.pseudoknot_nucleotides(pairs_torch)
    assert pk_torch.tolist() == [1, 3]

    empty = pseudoknot.pseudoknot_nucleotides(np.empty((0, 2), dtype=int))
    assert empty.size == 0


def test_pseudoknot_nucleotides_list() -> None:
    pk_list = pseudoknot.pseudoknot_nucleotides([[0, 2], [1, 3]])
    assert pk_list == [1, 3]


def test_normalize_pairs_numpy_and_torch() -> None:
    pairs_np = np.array([[3, 1], [1, 3], [2, 0]], dtype=int)
    norm_np = pseudoknot._numpy_normalize_pairs(pairs_np)
    assert norm_np.tolist() == [[0, 2], [1, 3]]

    with pytest.raises(TypeError, match="shape"):
        pseudoknot._numpy_normalize_pairs(np.array([1, 2, 3]))

    norm_empty = pseudoknot._numpy_normalize_pairs(np.empty((0, 2), dtype=int))
    assert norm_empty.shape == (0, 2)

    pairs_torch = torch.tensor([[3, 1], [1, 3], [2, 0]], dtype=torch.long)
    norm_torch = pseudoknot._torch_normalize_pairs(pairs_torch)
    assert norm_torch.tolist() == [[0, 2], [1, 3]]

    with pytest.raises(TypeError, match="shape"):
        pseudoknot._torch_normalize_pairs(torch.tensor([1, 2, 3]))

    norm_empty_torch = pseudoknot._torch_normalize_pairs(torch.empty((0, 2), dtype=torch.long))
    assert norm_empty_torch.shape == (0, 2)


def test_torch_normalize_pairs_sorted_variants() -> None:
    pairs = torch.tensor([[0, 2], [0, 2], [1, 3]], dtype=torch.long)
    norm = pseudoknot._torch_normalize_pairs(pairs)
    assert norm.tolist() == [[0, 2], [1, 3]]

    pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    norm = pseudoknot._torch_normalize_pairs(pairs)
    assert norm.tolist() == [[0, 2], [1, 3]]


def test_crossing_mask_small_and_unsorted_numpy() -> None:
    pairs = np.array([[0, 4], [2, 3], [1, 5]], dtype=int)
    mask = pseudoknot._numpy_crossing_mask(pairs)
    assert mask.tolist() == [True, False, True]

    single = pseudoknot._numpy_crossing_mask(np.array([[0, 2]], dtype=int))
    assert single.tolist() == [False]

    empty = pseudoknot._numpy_crossing_mask(np.empty((0, 2), dtype=int))
    assert empty.size == 0


def test_crossing_mask_small_torch() -> None:
    pairs = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.long)
    mask = pseudoknot._torch_crossing_mask(pairs)
    assert mask.tolist() == [True, True, True]

    single = pseudoknot._torch_crossing_mask(torch.tensor([[0, 2]], dtype=torch.long))
    assert single.tolist() == [False]


def test_crossing_pairs_empty_tensor() -> None:
    empty = torch.empty((0, 2), dtype=torch.long)
    out = pseudoknot.crossing_pairs(empty)
    assert out.shape == (0, 2)


def test_crossing_mask_large_paths() -> None:
    count = pseudoknot._CROSSING_N2_THRESHOLD + 1
    base_pairs = [(0, 3), (1, 4)] + [(idx, idx + 1) for idx in range(2, count)]
    pairs_np = np.array(base_pairs, dtype=int)
    mask_np = pseudoknot._numpy_crossing_mask(pairs_np)
    assert mask_np[0]
    assert mask_np[1]
    assert not mask_np[2:].any()

    pairs_torch = torch.tensor(base_pairs, dtype=torch.long)
    mask_torch = pseudoknot._torch_crossing_mask(pairs_torch)
    assert bool(mask_torch[0].item())
    assert bool(mask_torch[1].item())
    assert not bool(mask_torch[2:].any().item())


def test_torch_crossing_mask_large_edge_cases() -> None:
    empty = torch.empty((0, 2), dtype=torch.long)
    mask = pseudoknot._torch_crossing_mask_large(empty)
    assert mask.numel() == 0

    single = pseudoknot._torch_crossing_mask_large(torch.tensor([[0, 0]], dtype=torch.long))
    assert single.tolist() == [False]


def test_torch_range_max_and_mask_helpers() -> None:
    start = torch.tensor([], dtype=torch.long)
    l = torch.tensor([0], dtype=torch.long)  # noqa: E741
    r = torch.tensor([0], dtype=torch.long)
    out = pseudoknot._torch_range_max(start, l, r)  # noqa: E741
    assert out.tolist() == [-1]

    start = torch.tensor([3, 1, 4, 2], dtype=torch.long)
    l = torch.tensor([0, 2], dtype=torch.long)  # noqa: E741
    r = torch.tensor([2, 1], dtype=torch.long)
    out = pseudoknot._torch_range_max(start, l, r)
    assert out.tolist() == [4, -1]

    ii = torch.tensor([0, 1], dtype=torch.long)
    jj = torch.tensor([3, 4], dtype=torch.long)
    mask = pseudoknot._torch_crossing_mask_range_max(ii, jj, 5)
    assert mask.tolist() == [True, True]

    empty_mask = pseudoknot._torch_crossing_mask_range_max(
        torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), 3
    )
    assert empty_mask.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_torch_crossing_mask_cuda_small() -> None:
    pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long, device="cuda")
    mask = pseudoknot._torch_crossing_mask(pairs)
    assert mask.tolist() == [True, True]
