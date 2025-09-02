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

PAIR_BACKENDS = ("list", "numpy", "torch")
ARRAY_BACKENDS = ("numpy", "torch")
CROSSING_PAIRS = [(0, 2), (1, 3)]
DOT_BRACKET_ERROR_CASES = [
    pytest.param(")", "Unmatched symbol", id="unmatched_close"),
    pytest.param("1", "Invalid symbol", id="invalid_symbol"),
    pytest.param("(", "Unmatched symbol", id="unmatched_open"),
]


@pytest.fixture(params=PAIR_BACKENDS, ids=PAIR_BACKENDS)
def backend(request) -> str:
    return request.param


@pytest.fixture(params=ARRAY_BACKENDS, ids=ARRAY_BACKENDS)
def array_backend(request) -> str:
    return request.param


def as_list(values) -> list:
    return values.tolist() if hasattr(values, "tolist") else list(values)


def as_tuple_list(values) -> list[tuple]:
    return [tuple(item) for item in as_list(values)]


def as_nested_tuple_lists(values) -> list[list[tuple]]:
    return [as_tuple_list(item) for item in values]


def as_set(values) -> set[tuple]:
    return set(as_tuple_list(values))


def as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def make_pairs(pairs, backend: str):
    if not pairs:
        if backend == "list":
            return []
        if backend == "numpy":
            return np.empty((0, 2), dtype=int)
        if backend == "torch":
            return torch.empty((0, 2), dtype=torch.long)
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "list":
        return [tuple(pair) for pair in pairs]
    if backend == "numpy":
        return np.array(pairs, dtype=int)
    if backend == "torch":
        return torch.tensor(pairs, dtype=torch.long)
    raise ValueError(f"Unknown backend: {backend}")


def make_indices(values, backend: str):
    if backend == "list":
        return list(values)
    if backend == "numpy":
        return np.array(values, dtype=int)
    if backend == "torch":
        return torch.tensor(values, dtype=torch.long)
    raise ValueError(f"Unknown backend: {backend}")


def make_contact_map(data, backend: str, dtype=None):
    if backend == "list":
        return [list(row) for row in data]
    if backend == "numpy":
        if isinstance(dtype, torch.dtype):
            if dtype == torch.bool:
                dtype = bool
            elif dtype.is_floating_point:
                dtype = float
            elif dtype in (torch.int32, torch.int64):
                dtype = int
        return np.array(data, dtype=dtype)
    if backend == "torch":
        if dtype is bool:
            dtype = torch.bool
        elif dtype is float:
            dtype = torch.float
        elif dtype is int:
            dtype = torch.long
        return torch.tensor(data, dtype=dtype)
    raise ValueError(f"Unknown backend: {backend}")
