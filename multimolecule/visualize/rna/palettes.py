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

from contextlib import suppress
from typing import Mapping

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from ..palettes import DEFAULT_PALETTE, PALETTES, map_color_slots, resolve_palette

NUCLEOTIDE_COLOR_SLOTS: dict[str, str] = {
    "A": "blue",
    "C": "green",
    "G": "yellow",
    "U": "red",
    "T": "red",
    "N": "gray",
}

LOOP_TYPE_COLOR_SLOTS: dict[str, str] = {
    "stem": "blue",
    "hairpin": "red",
    "bulge": "orange",
    "internal": "green",
    "multiloop": "purple",
    "external": "gray",
    "end": "gray",
}

PAIR_COMPARISON_COLOR_SLOTS: dict[str, str] = {
    "true_positive": "green",
    "false_positive": "red",
    "false_negative": "gray",
}

_STRUCTURAL_CLASS_TO_LOOP_TYPE: dict[str, str] = {
    "S": "stem",
    "H": "hairpin",
    "B": "bulge",
    "I": "internal",
    "M": "multiloop",
    "X": "external",
    "E": "end",
}

PSEUDOKNOT_TIER_COLOR_SLOTS: tuple[str, ...] = ("red", "blue", "green", "purple", "yellow")

PAIR_PROBABILITY_CMAP = "multimolecule_probability"
PAIR_PROBABILITY_COLORS: tuple[str, ...] = (
    "#f7fbff",
    "#c8e7fb",
    "#9dcbec",
    "#5799d1",
    "#0272b2",
    "#012c5c",
)


def resolve_nucleotide_colors(palette: str, base_colors: Mapping[str, str] | None = None) -> dict[str, str]:
    colors = map_color_slots(NUCLEOTIDE_COLOR_SLOTS, palette)
    if base_colors:
        colors.update({base.upper(): color for base, color in base_colors.items()})
    return colors


def _register_pair_probability_cmap() -> None:
    cmap = LinearSegmentedColormap.from_list(
        PAIR_PROBABILITY_CMAP,
        list(PAIR_PROBABILITY_COLORS),
    )
    with suppress(ValueError, AttributeError):
        mpl.colormaps.register(cmap)


NUCLEOTIDE_COLORS: dict[str, str] = resolve_nucleotide_colors(DEFAULT_PALETTE)
NUCLEOTIDE_PALETTES: dict[str, dict[str, str]] = {name: resolve_nucleotide_colors(name) for name in PALETTES}
LOOP_TYPE_PALETTE: dict[str, str] = map_color_slots(LOOP_TYPE_COLOR_SLOTS, DEFAULT_PALETTE)
PAIR_COMPARISON_COLORS: dict[str, str] = map_color_slots(PAIR_COMPARISON_COLOR_SLOTS, DEFAULT_PALETTE)
PSEUDOKNOT_TIER_COLORS: tuple[str, ...] = tuple(
    resolve_palette(DEFAULT_PALETTE).get(slot, slot) for slot in PSEUDOKNOT_TIER_COLOR_SLOTS
)
STRUCTURAL_CLASS_PALETTE: dict[str, str] = {
    structural_class: LOOP_TYPE_PALETTE[loop_type]
    for structural_class, loop_type in _STRUCTURAL_CLASS_TO_LOOP_TYPE.items()
}

_register_pair_probability_cmap()
