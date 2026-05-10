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

import numpy as np

from multimolecule.visualization.rna import secondary_structure_layout


def test_layout_returns_per_position_arrays(nested_structure) -> None:
    _, dbn = nested_structure
    xs, ys = secondary_structure_layout(dbn)
    assert xs.shape == (len(dbn),)
    assert ys.shape == (len(dbn),)
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(ys))


def test_layout_half_span_bounds_coordinates(nested_structure) -> None:
    _, dbn = nested_structure
    xs, ys = secondary_structure_layout(dbn, half_span=1.0)
    assert xs.max() - xs.min() <= 2.0 + 1e-6
    assert ys.max() - ys.min() <= 2.0 + 1e-6


def test_layout_rotation_changes_coords(nested_structure) -> None:
    _, dbn = nested_structure
    xs_0, ys_0 = secondary_structure_layout(dbn, rotate=0)
    xs_90, ys_90 = secondary_structure_layout(dbn, rotate=90)
    assert not np.allclose(xs_0, xs_90) or not np.allclose(ys_0, ys_90)


def test_empty_layout_handled() -> None:
    xs, ys = secondary_structure_layout("")
    assert xs.shape == (0,)
    assert ys.shape == (0,)
