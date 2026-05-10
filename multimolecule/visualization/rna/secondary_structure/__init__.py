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

from ..palettes import (
    LOOP_TYPE_PALETTE,
    NUCLEOTIDE_COLORS,
    NUCLEOTIDE_PALETTES,
    PAIR_COMPARISON_COLORS,
    PAIR_PROBABILITY_CMAP,
    PAIR_PROBABILITY_COLORS,
    PSEUDOKNOT_TIER_COLORS,
    STRUCTURAL_CLASS_PALETTE,
)
from .alignment import plot_alignment_arcs
from .arc import plot_arc_diagram
from .circular import plot_circular_diagram
from .comparison import compare_secondary_structures
from .contact_map import plot_contact_map
from .layout import secondary_structure_layout
from .planar import plot_planar_graph
from .tracks import (
    BaseCategoryTrack,
    BaseValueTrack,
    LeontisWesthof,
    PairAnnotationTrack,
    PairProbabilityTrack,
    RegionTrack,
    SequenceDiffTrack,
    Track,
)

__all__ = [
    "BaseCategoryTrack",
    "BaseValueTrack",
    "LOOP_TYPE_PALETTE",
    "LeontisWesthof",
    "NUCLEOTIDE_COLORS",
    "NUCLEOTIDE_PALETTES",
    "PAIR_COMPARISON_COLORS",
    "PAIR_PROBABILITY_CMAP",
    "PAIR_PROBABILITY_COLORS",
    "PSEUDOKNOT_TIER_COLORS",
    "PairAnnotationTrack",
    "PairProbabilityTrack",
    "RegionTrack",
    "STRUCTURAL_CLASS_PALETTE",
    "SequenceDiffTrack",
    "Track",
    "compare_secondary_structures",
    "plot_alignment_arcs",
    "plot_arc_diagram",
    "plot_circular_diagram",
    "plot_contact_map",
    "plot_planar_graph",
    "secondary_structure_layout",
]
