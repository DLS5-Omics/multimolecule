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

"""
Visualization utilities for MultiMolecule.

The core plotting backend is Matplotlib because it produces publication-oriented static figures and supports
SVG/PDF export. Interactive backends such as Plotly should live behind optional adapters instead of becoming a
required dependency of the core visualization namespace.
"""

from .palettes import (
    DEFAULT_PALETTE,
    DEFAULT_PALETTE_VARIANT,
    PALETTES,
    map_color_slots,
    resolve_palette,
)

__all__ = [
    "DEFAULT_PALETTE",
    "DEFAULT_PALETTE_VARIANT",
    "PALETTES",
    "map_color_slots",
    "resolve_palette",
]
