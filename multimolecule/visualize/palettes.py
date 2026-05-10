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

from typing import Mapping

PALETTES: dict[str, dict[str, dict[str, str]]] = {
    "nature": {
        "lightest": {
            "stone": "#f7f5ef",
            "gray": "#e6e6ed",
            "red": "#fad0ce",
            "blue": "#c8e7fb",
            "yellow": "#fff1c3",
            "olive": "#f4f1b3",
            "green": "#d9e8c6",
            "teal": "#cce7ee",
            "purple": "#ebd6e9",
            "orange": "#fde0bd",
            "tan": "#f8e6d7",
        },
        "light": {
            "stone": "#e4e0ce",
            "gray": "#c8ceda",
            "red": "#eda4a7",
            "blue": "#9dcbec",
            "yellow": "#f6dc8a",
            "olive": "#dfdc67",
            "green": "#a0ca79",
            "teal": "#97d2d4",
            "purple": "#d3aad1",
            "orange": "#fbc07f",
            "tan": "#ddbda3",
        },
        "accent": {
            "stone": "#c6c2a6",
            "gray": "#99a3b4",
            "red": "#de6866",
            "blue": "#5799d1",
            "yellow": "#ebc850",
            "olive": "#c9c700",
            "green": "#62b347",
            "teal": "#4bbcbd",
            "purple": "#bb7cb4",
            "orange": "#f59a45",
            "tan": "#bf997d",
        },
        "dark": {
            "stone": "#a8a386",
            "gray": "#6f7b91",
            "red": "#c93e3f",
            "blue": "#0272b2",
            "yellow": "#cca02c",
            "olive": "#9ba415",
            "green": "#459434",
            "teal": "#019aa3",
            "purple": "#a84e94",
            "orange": "#ec6f00",
            "tan": "#936a57",
        },
        "darker": {
            "stone": "#888366",
            "gray": "#49566d",
            "red": "#9c2826",
            "blue": "#014e91",
            "yellow": "#9c7717",
            "olive": "#66771e",
            "green": "#227130",
            "teal": "#016879",
            "purple": "#792c74",
            "orange": "#b74f06",
            "tan": "#755040",
        },
        "darkest": {
            "stone": "#625f4a",
            "gray": "#253247",
            "red": "#741915",
            "blue": "#012c5c",
            "yellow": "#6b5513",
            "olive": "#36461a",
            "green": "#163b1c",
            "teal": "#00394e",
            "purple": "#481951",
            "orange": "#843200",
            "tan": "#442d1f",
        },
    },
    "okabe_ito": {
        "accent": {
            "black": "#000000",
            "gray": "#d9d9d9",
            "neutral": "#d9d9d9",
            "orange": "#e69f00",
            "cyan": "#56b4e9",
            "green": "#009e73",
            "yellow": "#f0e442",
            "blue": "#0072b2",
            "red": "#d55e00",
            "purple": "#cc79a7",
            "teal": "#009e73",
        },
    },
    "stanford": {
        "accent": {
            "red": "#8C1515",
            "green": "#175E54",
            "mint": "#279989",
            "olive": "#8F993E",
            "sage": "#6FA287",
            "blue": "#4298B5",
            "teal": "#007C92",
            "orange": "#E98300",
            "coral": "#E04F39",
            "yellow": "#FEDD5C",
            "purple": "#620059",
            "burgundy": "#651C32",
            "brown": "#5D4B3C",
            "gray": "#7F7776",
            "neutral": "#DAD7CB",
            "black": "#2E2D29",
        },
        "light": {
            "red": "#B83A4B",
            "green": "#2D716F",
            "mint": "#59B3A9",
            "olive": "#A6B168",
            "sage": "#8AB8A7",
            "blue": "#67AFD2",
            "teal": "#009AB4",
            "orange": "#F9A44A",
            "coral": "#F4795B",
            "yellow": "#FFE781",
            "purple": "#734675",
            "burgundy": "#7F2D48",
            "brown": "#766253",
            "gray": "#D4D1D1",
            "neutral": "#F4F4F4",
        },
        "dark": {
            "red": "#820000",
            "green": "#014240",
            "mint": "#017E7C",
            "olive": "#7A863B",
            "sage": "#417865",
            "blue": "#016895",
            "teal": "#006B81",
            "orange": "#D1660F",
            "coral": "#C74632",
            "yellow": "#FEC51D",
            "purple": "#350D36",
            "burgundy": "#42081B",
            "brown": "#2F2424",
            "gray": "#544948",
            "neutral": "#B6B1A9",
        },
    },
}
DEFAULT_PALETTE = "nature"
DEFAULT_PALETTE_VARIANT = "accent"


def resolve_palette(palette: str, variant: str = DEFAULT_PALETTE_VARIANT) -> dict[str, str]:
    try:
        variants = PALETTES[palette]
    except KeyError:
        choices = ", ".join(sorted(PALETTES))
        raise ValueError(f"Unknown palette: {palette}. Available palettes: {choices}.") from None
    try:
        return dict(variants[variant])
    except KeyError:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown palette variant: {variant}. Available variants for {palette}: {choices}.") from None


def map_color_slots(
    slots: Mapping[str, str], palette: str = DEFAULT_PALETTE, variant: str = DEFAULT_PALETTE_VARIANT
) -> dict[str, str]:
    colors = resolve_palette(palette, variant)
    return {key: colors.get(slot, slot) for key, slot in slots.items()}
