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


from .bprna import annotate, annotate_function, annotate_structure
from .noncanonical import noncanonical_pairs, noncanonical_pairs_set
from .notations import (
    contact_map_to_dot_bracket,
    contact_map_to_pairs,
    dot_bracket_to_contact_map,
    dot_bracket_to_pairs,
    pairs_to_contact_map,
    pairs_to_dot_bracket,
)
from .pairs import (
    Pair,
    ensure_pairs_list,
    ensure_pairs_np,
    normalize_pairs,
    pairs_to_duplex_segment_arrays,
    pairs_to_helix_segment_arrays,
    pairs_to_stem_segment_arrays,
    segment_arrays_to_pairs,
    sort_pairs,
)
from .pseudoknot import (
    crossing_arcs,
    crossing_events,
    crossing_mask,
    crossing_nucleotides,
    crossing_pairs,
    has_pseudoknot,
    nested_pairs,
    pseudoknot_nucleotides,
    pseudoknot_pairs,
    pseudoknot_tiers,
    split_crossing_pairs,
    split_pseudoknot_pairs,
)
from .topology import (
    HelixSegments,
    LoopHelixGraph,
    Loops,
    LoopSegments,
    RnaSecondaryStructureTopology,
    StemSegments,
    Tier,
)
from .types import (
    EdgeType,
    EndSide,
    HelixSegment,
    Loop,
    LoopContext,
    LoopRole,
    LoopSegment,
    LoopSegmentContext,
    LoopSegmentType,
    LoopSpan,
    LoopType,
    LoopView,
    PseudoknotType,
    StemEdge,
    StemEdgeType,
    StemSegment,
    StructureView,
)

__all__ = [
    "EdgeType",
    "EndSide",
    "HelixSegment",
    "HelixSegments",
    "Pair",
    "Loop",
    "LoopContext",
    "LoopSegment",
    "LoopSegmentContext",
    "LoopSegmentType",
    "LoopSegments",
    "LoopSpan",
    "Loops",
    "LoopView",
    "StructureView",
    "LoopRole",
    "LoopHelixGraph",
    "PseudoknotType",
    "Tier",
    "RnaSecondaryStructureTopology",
    "LoopType",
    "StemSegment",
    "StemSegments",
    "StemEdgeType",
    "StemEdge",
    "annotate",
    "annotate_function",
    "annotate_structure",
    "crossing_arcs",
    "crossing_events",
    "crossing_mask",
    "crossing_nucleotides",
    "crossing_pairs",
    "has_pseudoknot",
    "normalize_pairs",
    "nested_pairs",
    "sort_pairs",
    "pseudoknot_tiers",
    "split_crossing_pairs",
    "split_pseudoknot_pairs",
    "contact_map_to_dot_bracket",
    "contact_map_to_pairs",
    "dot_bracket_to_contact_map",
    "dot_bracket_to_pairs",
    "pairs_to_contact_map",
    "pairs_to_dot_bracket",
    "pseudoknot_nucleotides",
    "pseudoknot_pairs",
    "ensure_pairs_list",
    "ensure_pairs_np",
    "pairs_to_helix_segment_arrays",
    "pairs_to_duplex_segment_arrays",
    "pairs_to_stem_segment_arrays",
    "segment_arrays_to_pairs",
    "noncanonical_pairs",
    "noncanonical_pairs_set",
]
