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

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Sequence, Tuple

from torch import Tensor

try:
    from enum import StrEnum
except ImportError:
    from strenum import LowercaseStrEnum as StrEnum  # type: ignore[no-redef]

Pair = Tuple[int, int]
Pairs = Sequence[Pair]
PairsList = List[Pair]
Segment = PairsList
Tiers = List[PairsList]
Edge = Tuple[int, int]
Edges = Sequence[Edge]


class StructureView(StrEnum):
    ALL = "all"
    NESTED = "nested"
    PSEUDOKNOT = "pseudoknot"

    @classmethod
    def parse(cls, view: StructureView | str | None) -> StructureView:
        if view is None:
            return cls.ALL
        if isinstance(view, cls):
            return view
        return cls(view)


class LoopType(IntEnum):
    HAIRPIN = 1
    BULGE = 2
    INTERNAL = 3
    MULTILOOP = 4
    EXTERNAL = 5


class LoopView(StrEnum):
    Topological = auto()
    Nested = auto()
    Taxonomy = auto()

    @classmethod
    def parse(cls, mode: LoopView | str) -> LoopView:
        if isinstance(mode, cls):
            return mode
        return cls(mode)


@dataclass(frozen=True)
class LoopSpan:
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start + 1


class LoopRole(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    K1 = "K1"
    K2 = "K2"


class LoopSegmentType(IntEnum):
    HAIRPIN = 1
    BULGE = 2
    INTERNAL = 3
    BRANCH = 4
    EXTERNAL = 5
    END = 6


class EndSide(StrEnum):
    FIVE_PRIME = "5p"
    THREE_PRIME = "3p"


@dataclass(frozen=True)
class StemEdge:
    src: int
    dst: int
    src_pos: int
    dst_pos: int
    type: StemEdgeType
    tier: int


@dataclass(frozen=True, init=False)
class Loop:
    kind: LoopType
    spans: Tuple[LoopSpan, ...]
    anchor_pairs: Tuple[Pair, ...]
    anchor_helices: Tuple[HelixSegment, ...]
    anchor_tiers: Tuple[int, ...]
    is_external: bool
    role: LoopRole | None = None
    pseudoknot_type: PseudoknotType | None = None
    is_nested: bool = True

    def __init__(
        self,
        kind: LoopType,
        spans: Sequence[LoopSpan],
        anchor_pairs: Sequence[Pair],
        anchor_helices: Sequence[HelixSegment],
        anchor_tiers: Sequence[int],
        is_external: bool,
        *,
        role: LoopRole | None = None,
        pseudoknot_type: PseudoknotType | None = None,
        is_nested: bool = True,
    ) -> None:
        if pseudoknot_type is None:
            pseudoknot_type = PseudoknotType.NONE
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "spans", tuple(spans))
        object.__setattr__(self, "anchor_pairs", tuple(anchor_pairs))
        object.__setattr__(self, "anchor_helices", tuple(anchor_helices))
        object.__setattr__(self, "anchor_tiers", tuple(anchor_tiers))
        object.__setattr__(self, "is_external", is_external)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "pseudoknot_type", pseudoknot_type)
        object.__setattr__(self, "is_nested", is_nested)

    @property
    def size(self) -> int:
        return sum(len(span) for span in self.spans)

    @property
    def span_lengths(self) -> Tuple[int, ...]:
        return tuple(len(span) for span in self.spans)

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def anchor_helix_count(self) -> int:
        return len(self.anchor_helices)

    @property
    def branch_count(self) -> int:
        return self.anchor_helix_count

    @property
    def asymmetry(self) -> int:
        lengths = self.span_lengths
        if len(lengths) <= 1:
            return 0
        return max(lengths) - min(lengths)

    def overlaps_span(self, start: int, stop: int) -> bool:
        return any(span.start <= stop and span.stop >= start for span in self.spans)

    def with_taxonomy(
        self,
        pseudoknot_type: PseudoknotType,
        is_nested: bool,
        *,
        role: LoopRole | None = None,
    ) -> Loop:
        if role is None:
            role = self.role
        return Loop(
            self.kind,
            self.spans,
            self.anchor_pairs,
            self.anchor_helices,
            self.anchor_tiers,
            self.is_external,
            role=role,
            pseudoknot_type=pseudoknot_type,
            is_nested=is_nested,
        )

    def anchor_positions(self) -> Tuple[int, ...]:
        cached = getattr(self, "_anchor_positions_cache", None)
        if cached is not None:
            return cached
        spans = self.spans
        positions: list[int] = []
        for i, j in self.anchor_pairs:
            candidates: list[int] = []
            for span in spans:
                if i == span.start - 1 or i == span.stop + 1:
                    candidates.append(i)
                if j == span.start - 1 or j == span.stop + 1:
                    candidates.append(j)
            positions.append(min(candidates) if candidates else min(i, j))
        cached = tuple(positions)
        object.__setattr__(self, "_anchor_positions_cache", cached)
        return cached

    def ordered_anchor_pairs(self) -> Tuple[Pair, ...]:
        cached = getattr(self, "_ordered_anchor_pairs_cache", None)
        if cached is not None:
            return cached
        if not self.anchor_pairs:
            cached = ()
            object.__setattr__(self, "_ordered_anchor_pairs_cache", cached)
            return cached
        positions = self.anchor_positions()
        items = list(zip(positions, self.anchor_pairs))
        items.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
        cached = tuple(pair for _, pair in items)
        object.__setattr__(self, "_ordered_anchor_pairs_cache", cached)
        return cached

    def ordered_anchor_helices(self) -> Tuple[HelixSegment, ...]:
        cached = getattr(self, "_ordered_anchor_helices_cache", None)
        if cached is not None:
            return cached
        if not self.anchor_helices:
            cached = ()
            object.__setattr__(self, "_ordered_anchor_helices_cache", cached)
            return cached
        positions = self.anchor_positions()
        pair_positions = {tuple(sorted(pair)): pos for pos, pair in zip(positions, self.anchor_pairs)}
        items = []
        for helix in self.anchor_helices:
            anchor_pos = None
            for pair in helix.pairs:
                key = tuple(sorted(pair))
                if key in pair_positions:
                    pos = pair_positions[key]
                    anchor_pos = pos if anchor_pos is None else min(anchor_pos, pos)
            if anchor_pos is None:
                anchor_pos = min(helix.start_5p, helix.start_3p)
            items.append(
                (
                    anchor_pos,
                    helix.start_5p,
                    helix.start_3p,
                    helix.stop_5p,
                    helix.stop_3p,
                    helix,
                )
            )
        items.sort(key=lambda item: item[:5])
        cached = tuple(item[5] for item in items)
        object.__setattr__(self, "_ordered_anchor_helices_cache", cached)
        return cached


@dataclass(frozen=True)
class LoopSegment:
    kind: LoopSegmentType
    start: int
    stop: int
    tier: int = 0
    side: EndSide | None = None

    def __len__(self) -> int:
        return self.stop - self.start + 1


@dataclass(frozen=True)
class LoopSegmentContext:
    segment: LoopSegment
    pseudoknot_inside: Tensor
    pseudoknot_crossing: Tensor


@dataclass(frozen=True)
class LoopContext:
    loop: Loop
    pseudoknot_inside: Tensor
    pseudoknot_crossing: Tensor


@dataclass(frozen=True)
class StemSegment:
    start_5p: int
    stop_5p: int
    start_3p: int
    stop_3p: int
    tier: int

    def __len__(self) -> int:
        return self.stop_5p - self.start_5p + 1

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.start_5p, self.stop_5p, self.start_3p, self.stop_3p)


@dataclass(frozen=True)
class HelixSegment:
    start_5p: int
    stop_5p: int
    start_3p: int
    stop_3p: int
    pairs: Tuple[Pair, ...]
    tier: int

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.start_5p, self.stop_5p, self.start_3p, self.stop_3p)


class PseudoknotType(StrEnum):
    NONE = "none"
    H_TYPE = "h_type"
    KISSING_HAIRPIN = "kissing_hairpin"
    M_TYPE = "m_type"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class EdgeType(IntEnum):
    BACKBONE = 0
    NESTED_PAIRS = 1
    PSEUDOKNOT_PAIR = 2


class StemEdgeType(IntEnum):
    START_START = 1
    START_STOP = 2
    STOP_START = 3
    STOP_STOP = 4
