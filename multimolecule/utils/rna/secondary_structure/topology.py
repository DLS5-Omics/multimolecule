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

from bisect import bisect_right
from collections.abc import Sequence
from functools import cached_property
from typing import Dict, Generic, Iterator, List, Set, Tuple, TypeVar, overload

import numpy as np
import torch
from torch import Tensor

from ...graph import DirectedGraph, UndirectedGraph
from .noncanonical import noncanonical_pairs as noncanonical_pairs
from .notations import dot_bracket_to_pairs
from .pairs import (
    Pair,
    PairMap,
    Pairs,
    Segment,
    ensure_pairs_list,
    normalize_pairs,
    pairs_to_helix_segment_arrays,
    pairs_to_stem_segment_arrays,
    stem_segment_arrays_to_stem_segment_list,
)
from .pseudoknot import (
    crossing_arcs,
    crossing_events,
    crossing_nucleotides,
    crossing_pairs,
    pseudoknot_tiers,
    split_pseudoknot_pairs,
)
from .types import (
    Edge,
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

_PAIRS_TYPE_ERROR = "pairs must be a torch.Tensor or sequence of (i, j) pairs"

SegmentT = TypeVar("SegmentT", StemSegment, HelixSegment)


def compute_loop_spans(
    open_to_close: Tensor | Sequence[int],
    length: int,
) -> Tuple[List[int], List[List[int]], List[Tuple[int, int, LoopSegmentType, EndSide | None]]]:
    """
    Build loop intervals and opener/child relationships from an open-to-close map.

    This centralizes span construction so LoopSegments and bpRNA consumers
    share the same traversal.
    """
    if isinstance(open_to_close, Tensor):
        open_to_close_list = open_to_close.cpu().tolist()
    else:
        open_to_close_list = list(open_to_close)

    roots, children = LoopSegments._build_tree(open_to_close_list, length)
    intervals = LoopSegments._build_intervals(open_to_close_list, roots, children, length)
    return roots, children, intervals


class RnaSecondaryStructureTopology(UndirectedGraph):
    def __init__(
        self,
        sequence: str,
        secondary_structure: str | Tensor,
        device: torch.device | None = None,
        **kwargs,
    ):

        length = len(sequence)
        if isinstance(secondary_structure, torch.Tensor):
            pairs = secondary_structure
            if device is None:
                device = pairs.device
            elif pairs.device != device:
                pairs = pairs.to(device)
        elif isinstance(secondary_structure, str):
            if len(sequence) != len(secondary_structure):
                raise ValueError("sequence and secondary_structure must have the same length")
            pairs = torch.as_tensor(dot_bracket_to_pairs(secondary_structure), device=device)
            if device is None:
                device = pairs.device
        else:
            raise TypeError("secondary_structure must be a str or torch.Tensor")
        pairs = normalize_pairs(pairs)
        if device is None:
            device = pairs.device
        nested_pairs, pseudoknot_pairs = split_pseudoknot_pairs(pairs)

        if length > 1:
            backbone_idx = torch.arange(length - 1, dtype=torch.long, device=device)
            backbone_edges = torch.stack((backbone_idx, backbone_idx + 1), dim=1)
        else:
            backbone_edges = torch.empty((0, 2), dtype=torch.long, device=device)

        edge_index_parts: List[Tensor] = []
        edge_type_parts: List[Tensor] = []
        if len(backbone_edges):
            edge_index_parts.append(backbone_edges)
            edge_type_parts.append(
                torch.full((len(backbone_edges), 1), EdgeType.BACKBONE.value, dtype=torch.long, device=device)
            )
        if len(nested_pairs):
            edge_index_parts.append(nested_pairs)
            edge_type_parts.append(
                torch.full(
                    (len(nested_pairs), 1), EdgeType.NESTED_PAIRS.value, dtype=torch.long, device=nested_pairs.device
                )
            )
        if len(pseudoknot_pairs):
            edge_index_parts.append(pseudoknot_pairs)
            edge_type_parts.append(
                torch.full(
                    (len(pseudoknot_pairs), 1),
                    EdgeType.PSEUDOKNOT_PAIR.value,
                    dtype=torch.long,
                    device=pseudoknot_pairs.device,
                )
            )

        if edge_index_parts:
            edge_index = torch.cat(edge_index_parts, dim=0)
            edge_types = torch.cat(edge_type_parts, dim=0)
        else:
            edge_index = torch.empty((0, 2), dtype=torch.long, device=device)
            edge_types = torch.empty((0, 1), dtype=torch.long, device=device)

        self.sequence = sequence
        self.secondary_structure = secondary_structure
        self._all_pairs = pairs
        self._nested_pairs = nested_pairs
        self._pseudoknot_pairs = pseudoknot_pairs
        super().__init__(edge_index=edge_index, edge_features={"type": edge_types}, device=device, **kwargs)

    @property
    def all_pairs(self) -> Tensor:
        return self._all_pairs

    @property
    def nested_pairs(self) -> Tensor:
        return self._nested_pairs

    @property
    def pseudoknot_pairs(self) -> Tensor:
        return self._pseudoknot_pairs

    def pairs(self, view: StructureView | str | None = None) -> Tensor:
        view = StructureView.parse(view)
        if view == StructureView.ALL:
            return self._all_pairs
        if view == StructureView.NESTED:
            return self._nested_pairs
        if view == StructureView.PSEUDOKNOT:
            return self._pseudoknot_pairs
        raise ValueError("view must be 'all', 'nested', or 'pseudoknot'")

    @cached_property
    def crossing_events(self) -> Tensor:
        if self._all_pairs.shape[0] < 2:
            return self._all_pairs.new_empty((0, 2))
        return crossing_events(self._all_pairs)

    @cached_property
    def crossing_arcs(self) -> Tensor:
        if self._all_pairs.shape[0] < 2:
            return self._all_pairs.new_empty((0, 2))
        return crossing_arcs(self._all_pairs)

    @cached_property
    def crossing_pairs(self) -> Tensor:
        if self._all_pairs.shape[0] < 2:
            return self._all_pairs.new_empty((0, 2))
        return crossing_pairs(self._all_pairs)

    @cached_property
    def crossing_nucleotides(self) -> Tensor:
        if self._all_pairs.shape[0] < 2:
            return self._all_pairs.new_empty((0, 2))
        return crossing_nucleotides(self._all_pairs)

    @cached_property
    def tiers(self) -> List[Tier]:
        if len(self) == 0:
            return []
        pairs = self._all_pairs
        nested_pairs = self._nested_pairs
        pseudoknot_pairs = self._pseudoknot_pairs
        empty = pairs.new_empty((0, 2), dtype=torch.long)
        pair_tiers: List[Tensor] = [nested_pairs if nested_pairs.numel() else empty]
        if pseudoknot_pairs.numel() == 0:
            return [Tier(0, pair_tiers[0], len(self))]
        pk_tiers = pseudoknot_tiers(pseudoknot_pairs)
        for tier in pk_tiers:
            if isinstance(tier, Tensor):
                pair_tiers.append(tier)
            else:
                pair_tiers.append(torch.tensor(tier, dtype=torch.long, device=pairs.device))
        out: List[Tier] = []
        for idx, tier_pairs in enumerate(pair_tiers):
            if not isinstance(tier_pairs, Tensor):
                tier_pairs = torch.tensor(tier_pairs, dtype=torch.long, device=self._all_pairs.device)
            out.append(Tier(idx, tier_pairs, len(self)))
        return out

    @cached_property
    def nested_stem_segments(self) -> StemSegments:
        return StemSegments(self._nested_pairs, 0)

    @cached_property
    def all_stem_segments(self) -> StemSegments:
        return StemSegments(self._all_pairs, 0)

    @cached_property
    def nested_helix_segments(self) -> HelixSegments:
        return HelixSegments(self._nested_pairs, 0)

    @cached_property
    def all_helix_segments(self) -> HelixSegments:
        return HelixSegments(self._all_pairs, 0)

    @cached_property
    def nested_loop_segments(self) -> LoopSegments:
        return LoopSegments(self._nested_pairs, len(self), self._nested_pairs.device, 0)

    def loop_segment_contexts(
        self,
        level: int | None = None,
        *,
        view: StructureView | str | None = None,
    ) -> List[LoopSegmentContext]:
        if view is not None and level is not None:
            raise ValueError("view and level are mutually exclusive")
        if view is None:
            if level is None:
                return self.nested_loop_segments.contexts(self._pseudoknot_pairs)
            return self.tier_at(level).loop_segments.contexts(self._pseudoknot_pairs)
        view = StructureView.parse(view)
        if view == StructureView.NESTED:
            return self.nested_loop_segments.contexts(self._pseudoknot_pairs)
        raise ValueError("view must be 'nested' when requesting a single LoopSegments")

    def _tiers_for_view(self, view: StructureView | str | None) -> List[Tier]:
        view = StructureView.parse(view)
        if view == StructureView.ALL:
            return self.tiers
        if view == StructureView.NESTED:
            return self.tiers[:1]
        if view == StructureView.PSEUDOKNOT:
            return self.tiers[1:]
        raise ValueError("view must be 'all', 'nested', or 'pseudoknot'")

    def stems(self, view: StructureView | str | None = None) -> List[StemSegment]:
        tiers = self._tiers_for_view(view)
        return [segment for tier in tiers for segment in tier.stem_segments.segments]

    def helices(self, view: StructureView | str | None = None) -> List[HelixSegment]:
        tiers = self._tiers_for_view(view)
        return [segment for tier in tiers for segment in tier.helix_segments.segments]

    @cached_property
    def _partner_map_all(self) -> Tensor:
        return self._partner_map_from_pairs(self._all_pairs, len(self))

    @staticmethod
    def _partner_map_from_pairs(pairs: Tensor, length: int) -> Tensor:
        pair_map = pairs.new_full((length,), -1, dtype=torch.long)
        if pairs.numel() == 0:
            return pair_map
        left = torch.minimum(pairs[:, 0], pairs[:, 1]).to(torch.long)
        right = torch.maximum(pairs[:, 0], pairs[:, 1]).to(torch.long)
        pair_map[left] = right
        pair_map[right] = left
        return pair_map

    def partner_at(
        self,
        pos: int,
        level: int | None = None,
        *,
        view: StructureView | str | None = None,
    ) -> int | None:
        if pos >= len(self):
            raise IndexError("position is out of range")
        if view is not None and level is not None:
            raise ValueError("view and level are mutually exclusive")
        if view is not None:
            view = StructureView.parse(view)
            if view == StructureView.ALL:
                partner = int(self._partner_map_all[pos].item())
                return None if partner < 0 else partner
            pair_map = self._partner_map_from_pairs(self.pairs(view=view), len(self))
            partner = int(pair_map[pos].item())
            return None if partner < 0 else partner
        if level is None:
            partner = int(self._partner_map_all[pos].item())
            return None if partner < 0 else partner
        pair_map = self._partner_map_from_pairs(self.tier_at(level).pairs, len(self))
        partner = int(pair_map[pos].item())
        return None if partner < 0 else partner

    def paired_positions(self, view: StructureView | str | None = None) -> List[int]:
        view = StructureView.parse(view)
        pairs = self.pairs(view=view)
        if isinstance(pairs, Tensor):
            return StemSegments._paired_positions_from_pairs(pairs)
        pairs_list = ensure_pairs_list(pairs)
        if not pairs_list:
            return []
        positions = {i for i, j in pairs_list}
        positions.update(j for i, j in pairs_list)
        return sorted(positions)

    def unpaired_positions(self, view: StructureView | str | None = None) -> List[int]:
        view = StructureView.parse(view)
        if view == StructureView.ALL:
            pair_map = self._partner_map_all
        else:
            pair_map = self._partner_map_from_pairs(self.pairs(view=view), len(self))
        if pair_map.numel() == 0:
            return []
        return [idx for idx, partner in enumerate(pair_map.tolist()) if partner < 0]

    def noncanonical_pairs(
        self,
        view: StructureView | str | None = None,
        *,
        unsafe: bool = True,
    ) -> Tensor:
        pairs = self.pairs(view=view)
        return noncanonical_pairs(pairs, self.sequence, unsafe=unsafe)

    @cached_property
    def _stem_pair_map(self) -> Dict[Pair, StemSegment]:
        return StemSegments.pair_map_by_tiers(ensure_pairs_list(self._all_pairs))

    @cached_property
    def _helix_pair_map(self) -> Dict[Pair, HelixSegment]:
        mapping: Dict[Pair, HelixSegment] = {}
        for tier in self.tiers:
            mapping.update(tier.helix_segments.pair_map)
        return mapping

    def stem_at(
        self,
        pos: int,
        level: int | None = None,
        *,
        view: StructureView | str | None = None,
    ) -> StemSegment | None:
        if view is not None and level is not None:
            raise ValueError("view and level are mutually exclusive")
        if view is not None:
            view = StructureView.parse(view)
            partner = self.partner_at(pos, view=view)
            if partner is None:
                return None
            pair = (min(pos, partner), max(pos, partner))
            if view == StructureView.NESTED:
                return self.tier_at(0).stem_segments.pair_map.get(pair)
            if view == StructureView.PSEUDOKNOT:
                segment = self._stem_pair_map.get(pair)
                return segment if segment is not None and segment.tier > 0 else None
            return self._stem_pair_map.get(pair)
        partner = self.partner_at(pos, level=level)
        if partner is None:
            return None
        pair = (min(pos, partner), max(pos, partner))
        if level is None:
            return self._stem_pair_map.get(pair)
        return self.tier_at(level).stem_segments.pair_map.get(pair)

    def helix_at(
        self,
        pos: int,
        level: int | None = None,
        *,
        view: StructureView | str | None = None,
    ) -> HelixSegment | None:
        if view is not None and level is not None:
            raise ValueError("view and level are mutually exclusive")
        if view is not None:
            view = StructureView.parse(view)
            partner = self.partner_at(pos, view=view)
            if partner is None:
                return None
            pair = (min(pos, partner), max(pos, partner))
            if view == StructureView.NESTED:
                return self.tier_at(0).helix_segments.pair_map.get(pair)
            if view == StructureView.PSEUDOKNOT:
                segment = self._helix_pair_map.get(pair)
                return segment if segment is not None and segment.tier > 0 else None
            return self._helix_pair_map.get(pair)
        partner = self.partner_at(pos, level=level)
        if partner is None:
            return None
        pair = (min(pos, partner), max(pos, partner))
        if level is None:
            return self._helix_pair_map.get(pair)
        return self.tier_at(level).helix_segments.pair_map.get(pair)

    @staticmethod
    def _select_preferred_region(
        regions: Sequence[StemSegment | HelixSegment | LoopSegment],
        tier_preference: StructureView | str | None,
    ) -> StemSegment | HelixSegment | LoopSegment | None:
        if not regions:
            return None
        if tier_preference is None:
            return regions[0]
        preference = str(tier_preference).lower()
        if preference == "last":
            return regions[-1]
        if preference == "nested":
            return next((region for region in regions if region.tier == 0), None)
        if preference == "pseudoknot":
            return next((region for region in regions if region.tier > 0), None)
        return regions[0]

    def region_at(
        self,
        pos: int,
        *,
        view: StructureView | str | None = None,
        paired: str = "stem",
        tier_preference: StructureView | str | None = None,
        unpaired: str = "segment",
    ) -> StemSegment | HelixSegment | LoopSegment | Loop | None:
        if unpaired not in ("segment", "loop"):
            raise ValueError("unpaired must be 'segment' or 'loop'")
        if unpaired == "loop":
            if paired not in ("stem", "helix"):
                raise ValueError("paired must be 'stem' or 'helix'")
            selected_view = view
            if tier_preference is not None and selected_view is None:
                prefer_str = str(tier_preference).lower()
                if prefer_str == "nested":
                    selected_view = StructureView.NESTED
                elif prefer_str == "pseudoknot":
                    selected_view = StructureView.PSEUDOKNOT
            partner = self.partner_at(pos, view=selected_view)
            if partner is not None:
                if paired == "helix":
                    return self.helix_at(pos, view=selected_view)
                return self.stem_at(pos, view=selected_view)
            return self.loop_at(pos, view=selected_view)
        regions = self.regions_at(pos, view=view, paired=paired)
        if not regions:
            return None
        if tier_preference is None:
            for region in regions:
                if not isinstance(region, LoopSegment):
                    return region
            return regions[0]
        return self._select_preferred_region(regions, tier_preference)

    def regions_at(
        self,
        pos: int,
        *,
        view: StructureView | str | None = None,
        paired: str = "stem",
    ) -> List[StemSegment | HelixSegment | LoopSegment]:
        if pos >= len(self):
            raise IndexError("position is out of range")
        view = StructureView.parse(view)
        if paired not in ("stem", "helix"):
            raise ValueError("paired must be 'stem' or 'helix'")

        segments: List[StemSegment | HelixSegment | LoopSegment] = []
        for tier in self._tiers_for_view(view):
            pair_map = self._partner_map_from_pairs(tier.pairs, len(self))
            partner = int(pair_map[pos].item())
            segment: StemSegment | HelixSegment | LoopSegment | None = None
            if partner >= 0:
                pair = (min(pos, partner), max(pos, partner))
                if paired == "helix":
                    segment = tier.helix_segments.pair_map.get(pair)
                else:
                    segment = tier.stem_segments.pair_map.get(pair)
                if segment is not None:
                    segments.append(segment)
                continue
            segment = tier.loop_segments.segment_at(pos)
            if segment is not None:
                segments.append(segment)
        return segments

    def loop_at(
        self,
        pos: int,
        *,
        view: StructureView | str | None = None,
        mode: LoopView | str | None = None,
    ) -> Loop | None:
        if pos >= len(self):
            raise IndexError("position is out of range")
        return self.loops(view=view, mode=mode).loop_at(pos)

    def loop_segment_at(
        self,
        pos: int,
        *,
        view: StructureView | str | None = None,
        tier_preference: StructureView | str | None = None,
    ) -> LoopSegment | None:
        if pos >= len(self):
            raise IndexError("position is out of range")
        view = StructureView.parse(view)
        segments: List[LoopSegment] = []
        for tier in self._tiers_for_view(view):
            segment = tier.loop_segments.segment_at(pos)
            if segment is not None:
                segments.append(segment)
        selected = self._select_preferred_region(segments, tier_preference)
        return selected if isinstance(selected, LoopSegment) else None

    def loop_contexts(self) -> List[LoopContext]:
        """
        Return nested-loop contexts annotated with pseudoknot pairs.

        Loops are defined on nested pairs only; pseudoknot pairs are reported
        as inside or crossing each loop's spans.
        """
        loops = self.nested_loops
        pk_pairs = self._pseudoknot_pairs
        if not loops:
            return []
        if pk_pairs.numel() == 0:
            return [LoopContext(loop, pk_pairs, pk_pairs) for loop in loops]

        left = pk_pairs[:, 0].view(1, -1)
        right = pk_pairs[:, 1].view(1, -1)
        out: List[LoopContext] = []
        for loop in loops:
            if not loop.spans:
                empty = pk_pairs.new_empty((0, 2))
                out.append(LoopContext(loop, empty, empty))
                continue
            starts = torch.tensor(
                [span.start for span in loop.spans],
                device=pk_pairs.device,
                dtype=pk_pairs.dtype,
            ).view(-1, 1)
            stops = torch.tensor(
                [span.stop for span in loop.spans],
                device=pk_pairs.device,
                dtype=pk_pairs.dtype,
            ).view(-1, 1)
            in_left = (left >= starts) & (left <= stops)
            in_right = (right >= starts) & (right <= stops)
            left_mask = in_left.any(dim=0)
            right_mask = in_right.any(dim=0)
            inside_mask = left_mask & right_mask
            crossing_mask = left_mask ^ right_mask
            out.append(LoopContext(loop, pk_pairs[inside_mask], pk_pairs[crossing_mask]))
        return out

    def _resolve_loop_inputs(
        self,
        *,
        view: StructureView | str | None,
        mode: LoopView | str | None,
        pairs: Tensor | np.ndarray | Pairs | None,
    ) -> Tuple[Tensor | np.ndarray | Pairs, LoopView]:
        if mode is None:
            mode = LoopView.Topological
        mode = LoopView.parse(mode)
        if pairs is None:
            if mode == LoopView.Nested:
                if view is None:
                    view = StructureView.NESTED
                else:
                    view = StructureView.parse(view)
                    if view != StructureView.NESTED:
                        raise ValueError("mode 'nested' requires view='nested'")
            pairs = self.pairs(view=view)
        elif view is not None:
            raise ValueError("view and pairs are mutually exclusive")
        return pairs, mode

    def tier_at(self, level: int) -> Tier:
        if level < 0 or level >= len(self.tiers):
            raise IndexError("level is out of range")
        return self.tiers[level]

    @cached_property
    def nested_loops(self) -> Loops:
        return Loops.from_pairs(self._nested_pairs, len(self))

    @cached_property
    def all_loops(self) -> Loops:
        return Loops.from_pairs(self._all_pairs, len(self))

    @cached_property
    def taxonomy_loops(self) -> Loops:
        helix_segments = self.helices(view=StructureView.ALL)
        pair_to_helix = HelixSegments._pair_map_for_segments(helix_segments)
        pairs_list = ensure_pairs_list(self._all_pairs)
        crossing = crossing_pairs(pairs_list)
        flat_helix_segments = self.all_helix_segments
        return Loops.taxonomy_from_pairs(
            pairs_list,
            len(self),
            pair_to_helix=pair_to_helix,
            crossing=crossing,
            helix_segments=flat_helix_segments.segments,
            helix_edges=flat_helix_segments.edges,
            nested_pairs=self._nested_pairs,
            pseudoknot_pairs=self._pseudoknot_pairs,
        )

    def loops(
        self,
        view: StructureView | str | None = None,
        *,
        pairs: Tensor | np.ndarray | Pairs | None = None,
        mode: LoopView | str | None = None,
    ) -> Loops:
        pairs, mode = self._resolve_loop_inputs(view=view, mode=mode, pairs=pairs)
        if mode == LoopView.Taxonomy:
            return Loops.taxonomy_from_pairs(pairs, len(self))
        return Loops.from_pairs(pairs, len(self))

    def loop_helix_graph(
        self,
        pairs: Tensor | np.ndarray | Pairs | None = None,
        *,
        view: StructureView | str | None = None,
        mode: LoopView | str | None = None,
        level: int | None = None,
    ) -> LoopHelixGraph:
        pairs, mode = self._resolve_loop_inputs(view=view, mode=mode, pairs=pairs)
        pairs = normalize_pairs(pairs)
        pairs_list = ensure_pairs_list(pairs)
        device = pairs.device if isinstance(pairs, Tensor) else None
        if device is None:
            device = self._all_pairs.device
        if mode == LoopView.Taxonomy:
            loops = Loops.taxonomy_from_pairs(pairs, len(self))
        else:
            loops = Loops.from_pairs(pairs, len(self))
        level_indices = None if level is None else (level,)
        helices = HelixSegments.segments_by_tier(pairs_list, tiers=level_indices, device=device)

        loop_count = len(loops)
        loop_nodes = tuple(range(loop_count))
        helix_nodes = tuple(range(loop_count, loop_count + len(helices)))
        graph = UndirectedGraph(device=device)
        graph.add_nodes(loop_nodes)
        graph.add_nodes(helix_nodes)

        helix_key_to_idx = {(*helix.key, helix.tier): idx for idx, helix in enumerate(helices)}
        for loop_idx, loop in enumerate(loops):
            for helix in loop.anchor_helices:
                key = (*helix.key, helix.tier)
                helix_idx = helix_key_to_idx.get(key)
                if helix_idx is None:
                    continue
                graph.add_edge(loop_idx, loop_count + helix_idx, attr={"tier": helix.tier})

        return LoopHelixGraph(graph, loops, tuple(helices), loop_nodes, helix_nodes)

    def loop_span_sequences(self, loop: Loop) -> Tuple[str, ...]:
        return tuple(self.sequence[span.start : span.stop + 1] for span in loop.spans)

    def loop_sequence(self, loop: Loop, *, joiner: str = "") -> str:
        return joiner.join(self.loop_span_sequences(loop))

    def loop_anchor_pair_sequences(self, loop: Loop) -> Tuple[str, ...]:
        return tuple(self.sequence[i] + self.sequence[j] for i, j in loop.anchor_pairs)

    def stem_strands(self, stem: StemSegment, *, orientation: str = "5p-3p") -> Tuple[str, str]:
        return self._duplex_strands(stem.start_5p, stem.stop_5p, stem.start_3p, stem.stop_3p, orientation=orientation)

    def helix_strands(self, helix: HelixSegment, *, orientation: str = "5p-3p") -> Tuple[str, str]:
        return self._duplex_strands(
            helix.start_5p,
            helix.stop_5p,
            helix.start_3p,
            helix.stop_3p,
            orientation=orientation,
        )

    def _duplex_strands(
        self,
        start_5p: int,
        stop_5p: int,
        start_3p: int,
        stop_3p: int,
        *,
        orientation: str,
    ) -> Tuple[str, str]:
        if orientation not in ("5p-3p", "index"):
            raise ValueError("orientation must be '5p-3p' or 'index'")
        strand_5p = self.sequence[start_5p : stop_5p + 1]
        strand_3p = self.sequence[stop_3p : start_3p + 1]
        if orientation == "5p-3p":
            strand_3p = strand_3p[::-1]
        return strand_5p, strand_3p

    def annotate_positions(
        self,
        *,
        view: StructureView | str | None = None,
        paired: str = "stem",
        mode: LoopView | str | None = None,
        tier_preference: StructureView | str | None = None,
    ) -> List[Dict[str, object]]:
        """
        Return per-position annotations for paired/loop context.

        Each entry includes partner, paired flag, tier, segment/loop objects,
        and indices where available. Negative indexing follows normal Python rules.
        """
        if paired not in ("stem", "helix"):
            raise ValueError("paired must be 'stem' or 'helix'")
        view = StructureView.parse(view)
        loops = self.loops(view=view, mode=mode)
        loop_index_map = loops._position_to_loop_index
        segments = self.helices(view=view) if paired == "helix" else self.stems(view=view)
        segment_index_map = {segment: idx for idx, segment in enumerate(segments)}

        annotations: List[Dict[str, object]] = []
        for pos in range(len(self)):
            partner = self.partner_at(pos, view=view)
            is_paired = partner is not None
            segment: StemSegment | HelixSegment | None = None
            segment_index: int | None = None
            segment_type: str | None = None
            loop_segment: LoopSegment | None = None
            loop: Loop | None = None
            loop_index: int | None = None
            loop_kind: LoopType | None = None
            tier: int | None = None

            if is_paired:
                if paired == "helix":
                    segment = self.helix_at(pos, view=view)
                    segment_type = "helix"
                else:
                    segment = self.stem_at(pos, view=view)
                    segment_type = "stem"
                if segment is not None:
                    segment_index = segment_index_map.get(segment)
                    tier = segment.tier
            else:
                loop_segment = self.loop_segment_at(pos, view=view, tier_preference=tier_preference)
                if loop_segment is not None:
                    tier = loop_segment.tier
                if pos < len(loop_index_map):
                    idx = loop_index_map[pos]
                    if idx != -1:
                        loop_index = idx
                        loop = loops[idx]
                        loop_kind = loop.kind

            annotations.append(
                {
                    "pos": pos,
                    "partner": partner,
                    "paired": is_paired,
                    "tier": tier,
                    "segment_type": segment_type,
                    "segment_index": segment_index,
                    "segment": segment,
                    "loop_segment": loop_segment,
                    "loop_index": loop_index,
                    "loop": loop,
                    "loop_kind": loop_kind,
                }
            )
        return annotations

    def __repr__(self) -> str:
        parts = [
            f"length={len(self)}",
            f"pairs={int(self._all_pairs.shape[0])}",
            f"nested={int(self._nested_pairs.shape[0])}",
            f"pseudoknot={int(self._pseudoknot_pairs.shape[0])}",
        ]
        tiers = self.__dict__.get("tiers")
        if tiers is not None:
            parts.append(f"tiers={len(tiers)}")
        parts.append(f"device='{self._all_pairs.device}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __len__(self) -> int:
        return len(self.sequence)


class LoopHelixGraph:
    def __init__(
        self,
        graph: UndirectedGraph,
        loops: Loops,
        helices: Sequence[HelixSegment],
        loop_nodes: Sequence[int],
        helix_nodes: Sequence[int],
    ) -> None:
        self.graph = graph
        self.loops = loops
        self.helices = tuple(helices)
        self.loop_nodes = tuple(loop_nodes)
        self.helix_nodes = tuple(helix_nodes)

    def __repr__(self) -> str:
        edge_count = int(self.graph.edge_index.shape[0])
        parts = [
            f"loops={len(self.loops)}",
            f"helices={len(self.helices)}",
            f"edges={edge_count}",
        ]
        if self.helices:
            tiers = tuple(sorted({helix.tier for helix in self.helices}))
            parts.append(f"tiers={tiers}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def loop_helix_indices(self, loop_idx: int) -> Tuple[int, ...]:
        loop_node = self.loop_nodes[loop_idx]
        return self._neighbor_indices(loop_node, loop_nodes=False)

    def loop_neighbor_nodes(self, loop_idx: int) -> Tuple[int, ...]:
        return self._neighbors(self.loop_nodes[loop_idx])

    def helix_loop_indices(self, helix_idx: int) -> Tuple[int, ...]:
        helix_node = self.helix_nodes[helix_idx]
        return self._neighbor_indices(helix_node, loop_nodes=True)

    def helix_neighbor_nodes(self, helix_idx: int) -> Tuple[int, ...]:
        return self._neighbors(self.helix_nodes[helix_idx])

    def loop_helices(self, loop_idx: int) -> Tuple[HelixSegment, ...]:
        return tuple(self.helices[idx] for idx in self.loop_helix_indices(loop_idx))

    def helix_loops(self, helix_idx: int) -> Tuple[Loop, ...]:
        return tuple(self.loops[idx] for idx in self.helix_loop_indices(helix_idx))

    def loop_helix_edges(self) -> List[Edge]:
        edge_list = self._edge_list()
        if not edge_list:
            return []
        edges: List[Edge] = []
        loop_count = len(self.loops)
        for u, v in edge_list:
            if u < loop_count <= v:
                edges.append((u, v - loop_count))
            elif v < loop_count <= u:
                edges.append((v, u - loop_count))
        return edges

    def edge_list(self) -> List[Edge]:
        return self._edge_list()

    def adjacency_matrix(self, dtype: torch.dtype = torch.bool) -> Tensor:
        node_count = len(self.loop_nodes) + len(self.helix_nodes)
        adj = torch.zeros((node_count, node_count), dtype=dtype, device=self.graph.edge_index.device)
        edge_index = self.graph.edge_index
        if edge_index.numel() == 0:
            return adj
        adj[edge_index[:, 0], edge_index[:, 1]] = True if dtype == torch.bool else 1
        adj[edge_index[:, 1], edge_index[:, 0]] = True if dtype == torch.bool else 1
        return adj

    def _neighbors(self, node: int) -> Tuple[int, ...]:
        edge_index = self.graph.edge_index
        if edge_index.numel() == 0:
            return ()
        mask0 = edge_index[:, 0] == node
        mask1 = edge_index[:, 1] == node
        if not torch.any(mask0).item() and not torch.any(mask1).item():
            return ()
        neighbors: List[int] = []
        if torch.any(mask0).item():
            neighbors.extend(edge_index[mask0][:, 1].tolist())
        if torch.any(mask1).item():
            neighbors.extend(edge_index[mask1][:, 0].tolist())
        if not neighbors:
            return ()
        return tuple(sorted({int(n) for n in neighbors}))

    def _neighbor_indices(self, node: int, *, loop_nodes: bool) -> Tuple[int, ...]:
        neighbors = self._neighbors(node)
        if not neighbors:
            return ()
        loop_count = len(self.loops)
        if loop_nodes:
            return tuple(n for n in neighbors if n < loop_count)
        return tuple(n - loop_count for n in neighbors if n >= loop_count)

    def _edge_list(self) -> List[Edge]:
        edge_index = self.graph.edge_index
        if edge_index.numel() == 0:
            return []
        return [(int(edge[0]), int(edge[1])) for edge in edge_index.tolist()]


class Tier:
    def __init__(self, level: int, pairs: Tensor, length: int):
        self.level = level
        self.pairs = pairs
        self.length = length

    def __repr__(self) -> str:
        pair_count = int(self.pairs.shape[0])
        return f"{self.__class__.__name__}(level={self.level}, pairs={pair_count}, length={self.length})"

    @cached_property
    def loop_segments(self) -> LoopSegments:
        return LoopSegments(self.pairs, self.length, self.pairs.device, self.level)

    @cached_property
    def stem_segments(self) -> StemSegments:
        return StemSegments(self.pairs, self.level)

    @cached_property
    def helix_segments(self) -> HelixSegments:
        return HelixSegments(self.pairs, self.level)


class Loops:
    def __init__(self, loops: List[Loop], length: int):
        self._loops = loops
        self._length = length

    def __repr__(self) -> str:
        count = len(self._loops)
        parts = [f"count={count}", f"length={self._length}"]
        kind_counts = {
            LoopType.HAIRPIN: 0,
            LoopType.BULGE: 0,
            LoopType.INTERNAL: 0,
            LoopType.MULTILOOP: 0,
            LoopType.EXTERNAL: 0,
        }
        pk_counts: Dict[str, int] = {}
        for loop in self._loops:
            kind_counts[loop.kind] += 1
            pk_value = loop.pseudoknot_type.value if loop.pseudoknot_type is not None else "none"
            pk_counts[pk_value] = pk_counts.get(pk_value, 0) + 1
        parts.append(f"hairpin={kind_counts[LoopType.HAIRPIN]}")
        parts.append(f"bulge={kind_counts[LoopType.BULGE]}")
        parts.append(f"internal={kind_counts[LoopType.INTERNAL]}")
        parts.append(f"multiloop={kind_counts[LoopType.MULTILOOP]}")
        parts.append(f"external={kind_counts[LoopType.EXTERNAL]}")
        if pk_counts and (len(pk_counts) > 1 or pk_counts.get("none", 0) != count):
            ordered_pk = {k: pk_counts[k] for k in sorted(pk_counts)}
            parts.append(f"pk={ordered_pk}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def from_pairs(
        cls,
        pairs: Tensor | np.ndarray | Pairs,
        length: int,
        *,
        pseudoknot_type: PseudoknotType | None = None,
        is_nested: bool | None = None,
    ) -> Loops:
        return cls._from_pairs_list(
            ensure_pairs_list(pairs),
            length,
            pseudoknot_type=pseudoknot_type,
            is_nested=is_nested,
        )

    @classmethod
    def _from_pairs_list(
        cls,
        pairs_list: List[Pair],
        length: int,
        *,
        pair_to_helix: Dict[Pair, HelixSegment] | None = None,
        has_crossing: bool | None = None,
        pseudoknot_type: PseudoknotType | None = None,
        is_nested: bool | None = None,
    ) -> Loops:
        if has_crossing is None:
            has_crossing = bool(crossing_pairs(pairs_list)) if pairs_list else False
        if pseudoknot_type is None:
            pseudoknot_type = PseudoknotType.NONE if not has_crossing else PseudoknotType.UNKNOWN
        if is_nested is None:
            is_nested = not has_crossing
        partner_by_pos = PairMap(pairs_list, length=length).to_list()
        face_groups = cls._loop_face_groups(length, pairs_list, partner_by_pos)

        if pair_to_helix is None:
            if not has_crossing:
                pairs_tensor = torch.tensor(pairs_list, dtype=torch.long)
                helix_segments = HelixSegments._from_pairs(pairs_tensor, 0)
                pair_to_helix = HelixSegments._pair_map_for_segments(helix_segments)
            else:
                pair_to_helix = cls._helix_pair_map_by_tiers(pairs_list)
        loops: List[Loop] = []
        for outer_pair, (segments, anchor_pairs) in face_groups.items():
            anchor_pairs_list = sorted(anchor_pairs)
            anchor_helices = cls._anchor_helices(anchor_pairs_list, pair_to_helix)
            anchor_tiers = tuple(sorted({segment.tier for segment in anchor_helices}))
            loop_spans = tuple(LoopSpan(start, stop) for start, stop in segments)
            kind = cls._classify_loop(outer_pair is None, len(loop_spans), len(anchor_helices))
            loop_anchor_pairs = tuple(anchor_pairs_list)
            loops.append(
                Loop(
                    kind,
                    loop_spans,
                    loop_anchor_pairs,
                    tuple(anchor_helices),
                    anchor_tiers,
                    outer_pair is None,
                    pseudoknot_type=pseudoknot_type,
                    is_nested=is_nested,
                )
            )
        return cls(cls._sorted_loops(loops, length), length)

    @classmethod
    def taxonomy_from_pairs(
        cls,
        pairs: Tensor | np.ndarray | Pairs,
        length: int,
        *,
        pair_to_helix: Dict[Pair, HelixSegment] | None = None,
        crossing: List[Pair] | None = None,
        helix_segments: List[HelixSegment] | None = None,
        helix_edges: List[StemEdge] | None = None,
        nested_pairs: Tensor | np.ndarray | Pairs | None = None,
        pseudoknot_pairs: Tensor | np.ndarray | Pairs | None = None,
    ) -> Loops:
        return cls._taxonomy_from_pairs_list(
            ensure_pairs_list(pairs),
            length,
            pair_to_helix=pair_to_helix,
            crossing=crossing,
            helix_segments=helix_segments,
            helix_edges=helix_edges,
            nested_pairs=nested_pairs,
            pseudoknot_pairs=pseudoknot_pairs,
        )

    @classmethod
    def _taxonomy_from_pairs_list(
        cls,
        pairs_list: List[Pair],
        length: int,
        *,
        pair_to_helix: Dict[Pair, HelixSegment] | None = None,
        crossing: List[Pair] | None = None,
        helix_segments: List[HelixSegment] | None = None,
        helix_edges: List[StemEdge] | None = None,
        nested_pairs: Tensor | np.ndarray | Pairs | None = None,
        pseudoknot_pairs: Tensor | np.ndarray | Pairs | None = None,
    ) -> Loops:
        pairs_list = ensure_pairs_list(normalize_pairs(pairs_list))
        if nested_pairs is not None and pseudoknot_pairs is not None:
            if isinstance(pseudoknot_pairs, Tensor):
                has_pseudoknot = pseudoknot_pairs.numel() > 0
            elif isinstance(pseudoknot_pairs, np.ndarray):
                has_pseudoknot = pseudoknot_pairs.size > 0
            else:
                has_pseudoknot = bool(pseudoknot_pairs)
            if not has_pseudoknot:
                return cls._from_pairs_list(
                    pairs_list,
                    length,
                    pair_to_helix=pair_to_helix,
                    has_crossing=False,
                    pseudoknot_type=PseudoknotType.NONE,
                    is_nested=True,
                )
        if crossing is None:
            crossing = crossing_pairs(pairs_list)
        has_crossing = bool(crossing)

        def _build_loops(pseudoknot_type: PseudoknotType, is_nested: bool) -> Loops:
            return cls._from_pairs_list(
                pairs_list,
                length,
                pair_to_helix=pair_to_helix,
                has_crossing=has_crossing,
                pseudoknot_type=pseudoknot_type,
                is_nested=is_nested,
            )

        pk_type, h_type_stems, kissing, pk_span = cls._classify_pseudoknot(
            pairs_list,
            length,
            crossing=crossing,
            helix_segments=helix_segments,
            helix_edges=helix_edges,
            nested_pairs=nested_pairs,
            pseudoknot_pairs=pseudoknot_pairs,
        )
        if pk_type == PseudoknotType.NONE:
            return _build_loops(PseudoknotType.NONE, True)
        if pk_type == PseudoknotType.KISSING_HAIRPIN and kissing is not None:
            return kissing
        if pk_type in (PseudoknotType.M_TYPE, PseudoknotType.COMPLEX):
            topological = _build_loops(PseudoknotType.NONE, True)
            if pk_span is None:
                return _build_loops(pk_type, False)
            pk_loops: List[Loop] = []
            for loop in topological:
                if loop.overlaps_span(pk_span[0], pk_span[1]):
                    pk_loops.append(loop.with_taxonomy(pk_type, False))
                else:
                    pk_loops.append(loop)
            return cls(cls._sorted_loops(pk_loops, length), length)
        if pk_type != PseudoknotType.H_TYPE or h_type_stems is None:
            return _build_loops(PseudoknotType.UNKNOWN, False)

        stem1, stem2 = h_type_stems
        span_map = cls._h_type_loop_spans(stem1, stem2)
        topological = _build_loops(PseudoknotType.NONE, True)
        pk_start = stem1.start_5p
        pk_stop = stem2.start_3p
        h_loops: List[Loop] = []
        for loop in topological:
            if loop.overlaps_span(pk_start, pk_stop):
                continue
            h_loops.append(loop)

        if pair_to_helix is None:
            pair_to_helix = cls._helix_pair_map_by_tiers(pairs_list)
        pair_to_stem = cls._pair_to_stem_map(pair_to_helix)
        partner_by_pos = PairMap(pairs_list, length=length).to_list()
        for role, loop_span in span_map.items():
            loop_anchor_pairs = tuple(
                sorted(cls._anchor_pairs([(loop_span.start, loop_span.stop)], partner_by_pos, length, None))
            )
            anchor_helices = tuple(cls._anchor_helices(loop_anchor_pairs, pair_to_helix))
            anchor_stems = {pair_to_stem[pair] for pair in loop_anchor_pairs if pair in pair_to_stem}
            anchor_tiers = tuple(sorted({segment.tier for segment in anchor_helices}))
            kind = cls._classify_loop(False, 1, len(anchor_stems))
            h_loops.append(
                Loop(
                    kind,
                    (loop_span,),
                    loop_anchor_pairs,
                    anchor_helices,
                    anchor_tiers,
                    False,
                    role=role,
                    pseudoknot_type=PseudoknotType.H_TYPE,
                    is_nested=True,
                )
            )
        return cls(cls._sorted_loops(h_loops, length), length)

    @cached_property
    def hairpin_loops(self) -> List[Loop]:
        return [loop for loop in self._loops if loop.kind == LoopType.HAIRPIN]

    @cached_property
    def bulge_loops(self) -> List[Loop]:
        return [loop for loop in self._loops if loop.kind == LoopType.BULGE]

    @cached_property
    def internal_loops(self) -> List[Loop]:
        return [loop for loop in self._loops if loop.kind == LoopType.INTERNAL]

    @cached_property
    def multi_loops(self) -> List[Loop]:
        return [loop for loop in self._loops if loop.kind == LoopType.MULTILOOP]

    @cached_property
    def external_loops(self) -> List[Loop]:
        return [loop for loop in self._loops if loop.kind == LoopType.EXTERNAL]

    def spans(self) -> List[LoopSpan]:
        return [span for loop in self._loops for span in loop.spans]

    def __iter__(self) -> Iterator[Loop]:
        return iter(self._loops)

    def __len__(self) -> int:
        return len(self._loops)

    @overload
    def __getitem__(self, idx: int) -> Loop: ...

    @overload
    def __getitem__(self, idx: slice) -> Loops: ...

    def __getitem__(self, idx: int | slice) -> Loop | Loops:
        if isinstance(idx, slice):
            return Loops(self._loops[idx], self._length)
        return self._loops[idx]

    def loop_at(self, pos: int) -> Loop | None:
        if pos >= self._length:
            raise IndexError("position is out of range")
        idx = self._position_to_loop_index[pos]
        return None if idx == -1 else self._loops[idx]

    @cached_property
    def _position_to_loop_index(self) -> List[int]:
        if not self._loops:
            if self._length is None:
                return []
            return [-1] * self._length
        max_stop = max(span.stop for loop in self._loops for span in loop.spans)
        size = max_stop + 1
        if self._length is not None and self._length > size:
            size = self._length
        mapping = torch.full((size,), -1, dtype=torch.long)
        for idx, loop in enumerate(self._loops):
            for span in loop.spans:
                if span.start <= span.stop:
                    mapping[span.start : span.stop + 1] = idx
        return mapping.tolist()

    @staticmethod
    def _loop_sort_key(loop: Loop, length: int) -> int:
        if not loop.spans:
            return length
        return min(span.start for span in loop.spans)

    @classmethod
    def _sorted_loops(cls, loops: List[Loop], length: int) -> List[Loop]:
        loops.sort(key=lambda loop: cls._loop_sort_key(loop, length))
        return loops

    @staticmethod
    def _outermost_pair_index(pairs: List[Pair], length: int) -> Tuple[List[int], List[int]]:
        max_j_by_i = [-1] * length
        for i, j in pairs:
            if i > j:
                i, j = j, i
            if j > max_j_by_i[i]:
                max_j_by_i[i] = j
        prefix_max: List[int] = []
        current = -1
        for j in max_j_by_i:
            if j > current:
                current = j
            prefix_max.append(current)
        return max_j_by_i, prefix_max

    @staticmethod
    def _outermost_pair_from_index(
        start: int,
        stop: int,
        max_j_by_i: List[int],
        prefix_max: List[int],
    ) -> Pair | None:
        if start <= 0:
            return None
        upper = start - 1
        idx = bisect_right(prefix_max, stop, 0, upper + 1)
        if idx > upper:
            return None
        j = max_j_by_i[idx]
        if j <= stop:
            return None
        return idx, j

    @staticmethod
    def _anchor_pairs(
        segments: List[Pair],
        partner_by_pos: List[int],
        length: int,
        outer_pair: Pair | None,
    ) -> Set[Pair]:
        closing: Set[Pair] = set()
        for start, stop in segments:
            if start > 0 and partner_by_pos[start - 1] != -1:
                i = start - 1
                j = partner_by_pos[i]
                if i > j:
                    i, j = j, i
                closing.add((i, j))
            if stop + 1 < length and partner_by_pos[stop + 1] != -1:
                i = stop + 1
                j = partner_by_pos[i]
                if i > j:
                    i, j = j, i
                closing.add((i, j))
        if outer_pair is not None:
            closing.add(outer_pair)
        return closing

    @staticmethod
    def _face_segments(
        length: int,
        pairs: List[Pair],
        partner_by_pos: List[int],
    ) -> Dict[int, List[Pair]]:
        if length == 0:
            return {}
        if length == 1:
            if partner_by_pos[0] == -1:
                return {0: [(0, 0)]}
            return {}
        edges = [(idx, idx + 1) for idx in range(length - 1)]
        if pairs:
            edges.extend((i, j) for i, j in pairs if j != i + 1)

        degree = [0] * length
        neighbor0 = [-1] * length
        neighbor1 = [-1] * length
        neighbor2 = [-1] * length
        for idx in range(length):
            prev_idx = idx - 1 if idx - 1 >= 0 else None
            next_idx = idx + 1 if idx + 1 < length else None
            pair = partner_by_pos[idx]
            neighbors: List[int] = []
            if pair == -1:
                if prev_idx is not None:
                    neighbors.append(prev_idx)
                if next_idx is not None:
                    neighbors.append(next_idx)
            elif idx < pair:
                if prev_idx is not None:
                    neighbors.append(prev_idx)
                neighbors.append(pair)
                if next_idx is not None:
                    neighbors.append(next_idx)
            else:
                if next_idx is not None:
                    neighbors.append(next_idx)
                neighbors.append(pair)
                if prev_idx is not None:
                    neighbors.append(prev_idx)
            deg = len(neighbors)
            degree[idx] = deg
            if deg:
                neighbor0[idx] = neighbors[0]
                if deg > 1:
                    neighbor1[idx] = neighbors[1]
                    if deg > 2:
                        neighbor2[idx] = neighbors[2]

        half_to_face: Dict[int, int] = {}
        face_count = 0
        for u, v in edges:
            for src, dst in ((u, v), (v, u)):
                key = src * length + dst
                if key in half_to_face:
                    continue
                face_id = face_count
                face_count += 1
                cur_src = src
                cur_dst = dst
                while True:
                    cur_key = cur_src * length + cur_dst
                    if cur_key in half_to_face:
                        break
                    half_to_face[cur_key] = face_id
                    deg = degree[cur_dst]
                    if deg == 0:
                        break
                    n0 = neighbor0[cur_dst]
                    if deg == 1:
                        next_dst = n0
                    elif deg == 2:
                        n1 = neighbor1[cur_dst]
                        if cur_src == n0:
                            next_dst = n1
                        elif cur_src == n1:
                            next_dst = n0
                        else:
                            break
                    else:
                        n1 = neighbor1[cur_dst]
                        n2 = neighbor2[cur_dst]
                        if cur_src == n0:
                            next_dst = n1
                        elif cur_src == n1:
                            next_dst = n2
                        elif cur_src == n2:
                            next_dst = n0
                        else:
                            break
                    cur_src, cur_dst = cur_dst, next_dst

        face_positions: Dict[int, List[int]] = {}
        for idx, partner in enumerate(partner_by_pos):
            if partner != -1:
                continue
            if idx < length - 1:
                edge = (idx, idx + 1)
            else:
                edge = (length - 2, length - 1)
            face_key = half_to_face.get(edge[0] * length + edge[1])
            if face_key is None:
                continue
            face_positions.setdefault(face_key, []).append(idx)

        face_segments: Dict[int, List[Pair]] = {}
        for face_idx, positions in face_positions.items():
            positions = sorted(set(positions))
            segments: List[Pair] = []
            seg_start: int | None = None
            seg_prev: int | None = None
            for pos in positions:
                if seg_start is None:
                    seg_start = pos
                    seg_prev = pos
                    continue
                if seg_prev is not None and pos == seg_prev + 1:
                    seg_prev = pos
                    continue
                if seg_start is not None and seg_prev is not None:
                    segments.append((seg_start, seg_prev))
                seg_start = pos
                seg_prev = pos
            if seg_start is not None and seg_prev is not None:
                segments.append((seg_start, seg_prev))
            face_segments[face_idx] = segments
        return face_segments

    @classmethod
    def _loop_face_groups(
        cls,
        length: int,
        pairs: List[Pair],
        partner_by_pos: List[int],
    ) -> Dict[Pair | None, Tuple[List[Pair], Set[Pair]]]:
        max_j_by_i, prefix_max = cls._outermost_pair_index(pairs, length)
        face_segments = cls._face_segments(length, pairs, partner_by_pos)

        groups: Dict[Pair | None, List[Pair]] = {}
        for segments in face_segments.values():
            for start, stop in segments:
                outer = cls._outermost_pair_from_index(start, stop, max_j_by_i, prefix_max)
                groups.setdefault(outer, []).append((start, stop))
        grouped: Dict[Pair | None, Tuple[List[Pair], Set[Pair]]] = {}
        for outer_pair, segments in groups.items():
            segments = sorted(segments)
            grouped[outer_pair] = (segments, cls._anchor_pairs(segments, partner_by_pos, length, outer_pair))
        return grouped

    @staticmethod
    def _classify_loop(is_external: bool, span_count: int, anchor_helix_count: int) -> LoopType:
        if is_external:
            return LoopType.EXTERNAL
        if anchor_helix_count <= 1:
            return LoopType.HAIRPIN
        if anchor_helix_count == 2:
            return LoopType.BULGE if span_count <= 1 else LoopType.INTERNAL
        return LoopType.MULTILOOP

    @staticmethod
    def _anchor_helices(
        anchor_pairs: Sequence[Pair],
        pair_to_helix: Dict[Pair, HelixSegment],
    ) -> List[HelixSegment]:
        if not anchor_pairs or not pair_to_helix:
            return []
        anchors = {pair_to_helix[pair] for pair in anchor_pairs if pair in pair_to_helix}
        return sorted(anchors, key=HelixSegments._segment_sort_key)

    @staticmethod
    def _pair_to_stem_map(pair_to_helix: Dict[Pair, HelixSegment]) -> Dict[Pair, StemSegment]:
        if not pair_to_helix:
            return {}
        pairs_by_tier: Dict[int, List[Pair]] = {}
        for pair, helix in pair_to_helix.items():
            pairs_by_tier.setdefault(helix.tier, []).append(pair)
        pair_to_stem: Dict[Pair, StemSegment] = {}
        for tier, tier_pairs in pairs_by_tier.items():
            start_i, start_j, lengths = pairs_to_stem_segment_arrays(tier_pairs)
            stems = stem_segment_arrays_to_stem_segment_list(start_i, start_j, lengths, tier=tier)
            pair_to_stem.update(StemSegments._pair_map_for_segments(stems))
        return pair_to_stem

    @classmethod
    def _classify_pseudoknot(
        cls,
        pairs: List[Pair],
        length: int,
        *,
        crossing: List[Pair] | None = None,
        helix_segments: List[HelixSegment] | None = None,
        helix_edges: List[StemEdge] | None = None,
        nested_pairs: Tensor | np.ndarray | Pairs | None = None,
        pseudoknot_pairs: Tensor | np.ndarray | Pairs | None = None,
    ) -> Tuple[PseudoknotType, Tuple[StemSegment, StemSegment] | None, Loops | None, Pair | None]:
        if not pairs:
            return PseudoknotType.NONE, None, None, None
        crossing = crossing if crossing is not None else crossing_pairs(pairs)
        if not crossing:
            return PseudoknotType.NONE, None, None, None
        h_type_stems = cls._h_type_pseudoknot_stems(pairs, crossing=crossing)
        if h_type_stems is not None:
            return PseudoknotType.H_TYPE, h_type_stems, None, None
        kissing_signature = cls._kissing_hairpin_signature(
            pairs,
            length,
            nested_pairs=nested_pairs,
            pseudoknot_pairs=pseudoknot_pairs,
        )
        kissing = cls._kissing_hairpin_loops(pairs, length, kissing_signature=kissing_signature)
        if kissing is not None:
            return PseudoknotType.KISSING_HAIRPIN, None, kissing, None
        if helix_segments is None:
            pairs_tensor = torch.tensor(pairs, dtype=torch.long)
            helix_segments = HelixSegments._from_pairs(pairs_tensor, 0)
        stem_pairs = cls._crossing_stem_pairs(pairs, stem_list=helix_segments, crossing=crossing)
        pk_type = cls._taxonomy_type_from_pairs(
            pairs,
            length,
            crossing=crossing,
            h_type_stems=h_type_stems,
            kissing_signature=kissing_signature,
            stem_pairs=stem_pairs,
            helix_segments=helix_segments,
            helix_edges=helix_edges,
        )
        if pk_type in (PseudoknotType.M_TYPE, PseudoknotType.COMPLEX):
            crossing_stems: Set[StemSegment | HelixSegment] = {stem for pair in stem_pairs for stem in pair}
            span = cls._pseudoknot_span_from_stems(crossing_stems)
            return pk_type, None, None, span
        return PseudoknotType.UNKNOWN, None, None, None

    @classmethod
    def _h_type_pseudoknot_stems(
        cls,
        pairs: List[Pair],
        *,
        crossing: List[Pair] | None = None,
        pair_to_stem: Dict[Pair, StemSegment] | None = None,
    ) -> Tuple[StemSegment, StemSegment] | None:
        crossing = crossing if crossing is not None else crossing_pairs(pairs)
        if not crossing:
            return None
        if pair_to_stem is None:
            start_i, start_j, lengths = pairs_to_stem_segment_arrays(pairs)
            stems = stem_segment_arrays_to_stem_segment_list(start_i, start_j, lengths, tier=0)
            pair_to_stem = {}
            for stem in stems:
                seg_len = len(stem)
                for offset in range(seg_len):
                    i = stem.start_5p + offset
                    j = stem.start_3p - offset
                    if i > j:
                        i, j = j, i
                    pair_to_stem[(int(i), int(j))] = stem
        crossing_stems = {pair_to_stem[pair] for pair in crossing if pair in pair_to_stem}
        if len(crossing_stems) != 2:
            return None
        stem1, stem2 = sorted(crossing_stems, key=lambda stem: stem.start_5p)
        if not (stem1.start_5p < stem2.start_5p < stem1.start_3p < stem2.start_3p):
            return None
        if stem1.stop_5p >= stem2.start_5p or stem2.stop_5p >= stem1.stop_3p:
            return None
        if stem1.start_3p >= stem2.stop_3p:
            return None
        return stem1, stem2

    @staticmethod
    def _h_type_loop_spans(stem1: StemSegment, stem2: StemSegment) -> Dict[LoopRole, LoopSpan]:
        spans: Dict[LoopRole, LoopSpan] = {}
        l1_start = stem1.stop_5p + 1
        l1_stop = stem2.start_5p - 1
        if l1_start <= l1_stop:
            spans[LoopRole.L1] = LoopSpan(l1_start, l1_stop)
        l2_start = stem2.stop_5p + 1
        l2_stop = stem1.stop_3p - 1
        if l2_start <= l2_stop:
            spans[LoopRole.L2] = LoopSpan(l2_start, l2_stop)
        l3_start = stem1.start_3p + 1
        l3_stop = stem2.stop_3p - 1
        if l3_start <= l3_stop:
            spans[LoopRole.L3] = LoopSpan(l3_start, l3_stop)
        return spans

    @classmethod
    def _helix_pair_map_by_tiers(cls, pairs: List[Pair]) -> Dict[Pair, HelixSegment]:
        if not pairs:
            return {}
        pairs_tensor = torch.tensor(pairs, dtype=torch.long)
        tiers = StemSegments._tier_pairs(pairs_tensor)
        mapping: Dict[Pair, HelixSegment] = {}
        for tier_idx, tier_pairs in enumerate(tiers):
            segments = HelixSegments._from_pairs(tier_pairs, tier_idx)
            mapping.update(HelixSegments._pair_map_for_segments(segments))
        return mapping

    @classmethod
    def _crossing_stem_pairs(
        cls,
        pairs: List[Pair],
        *,
        stem_list: List[HelixSegment] | None = None,
        crossing: List[Pair] | None = None,
        pair_to_stem: Dict[Pair, HelixSegment] | None = None,
    ) -> List[Tuple[HelixSegment, HelixSegment]]:
        if not pairs:
            return []
        crossing = crossing if crossing is not None else crossing_pairs(pairs)
        if not crossing:
            return []
        if stem_list is None:
            pairs_tensor = torch.tensor(pairs, dtype=torch.long)
            stem_list = HelixSegments._from_pairs(pairs_tensor, 0)
        if len(stem_list) < 2:
            return []
        if pair_to_stem is None:
            pair_to_stem = HelixSegments._pair_map_for_segments(stem_list)
        crossing_stems = {pair_to_stem[pair] for pair in crossing if pair in pair_to_stem}
        if len(crossing_stems) < 2:
            return []
        stem_list = list(crossing_stems)
        stem_list = sorted(
            stem_list, key=lambda stem: (stem.start_5p, stem.start_3p, stem.stop_5p, stem.stop_3p, stem.tier)
        )
        stem_pairs: Set[Tuple[HelixSegment, HelixSegment]] = set()
        for idx, stem in enumerate(stem_list):
            for other in stem_list[idx + 1 :]:
                if other.start_5p >= stem.start_3p:
                    break
                if stem.start_5p < other.start_5p < stem.start_3p < other.start_3p:
                    stem_pairs.add((stem, other))
        return list(stem_pairs)

    @classmethod
    def _taxonomy_type_from_pairs(
        cls,
        pairs: List[Pair],
        length: int,
        *,
        crossing: List[Pair] | None = None,
        h_type_stems: Tuple[StemSegment, StemSegment] | None = None,
        kissing_signature: Tuple[Loops | None, Tuple[int, int], int] | None = None,
        stem_pairs: List[Tuple[HelixSegment, HelixSegment]] | None = None,
        helix_segments: List[HelixSegment] | None = None,
        helix_edges: List[StemEdge] | None = None,
    ) -> PseudoknotType | None:
        if not pairs:
            return None
        crossing = crossing if crossing is not None else crossing_pairs(pairs)
        if not crossing:
            return PseudoknotType.NONE
        if h_type_stems is None:
            h_type_stems = cls._h_type_pseudoknot_stems(pairs, crossing=crossing)
        if h_type_stems is not None:
            return PseudoknotType.H_TYPE
        if kissing_signature is None:
            kissing_signature = cls._kissing_hairpin_signature(pairs, length)
        if kissing_signature is not None:
            _nested_loops, _loop_indices, helix_count = kissing_signature
            return PseudoknotType.KISSING_HAIRPIN if helix_count == 1 else PseudoknotType.COMPLEX
        pairs_tensor: Tensor | None = None
        if helix_segments is None:
            pairs_tensor = torch.tensor(pairs, dtype=torch.long)
            helix_segments = HelixSegments._from_pairs(pairs_tensor, 0)
        if stem_pairs is None:
            stem_pairs = cls._crossing_stem_pairs(pairs, stem_list=helix_segments, crossing=crossing)
        if not stem_pairs:
            return PseudoknotType.UNKNOWN
        pseudoknot_stems = {stem for pair in stem_pairs for stem in pair}
        if len(pseudoknot_stems) < 3:
            return PseudoknotType.UNKNOWN
        if not helix_segments:
            return PseudoknotType.UNKNOWN
        pseudoknot_indices = {idx for idx, stem in enumerate(helix_segments) if stem in pseudoknot_stems}
        if len(pseudoknot_indices) != len(pseudoknot_stems):
            return PseudoknotType.UNKNOWN
        if helix_edges is None:
            if pairs_tensor is None:
                pairs_tensor = torch.tensor(pairs, dtype=torch.long)
            paired_positions = StemSegments._paired_positions_from_pairs(pairs_tensor)
            _graph, helix_edges = StemSegments._build_stem_graph_from_segments(
                helix_segments,
                paired_positions,
                tier=0,
                device=pairs_tensor.device,
                stop_endpoint="5p",
                stop_target="3p",
            )
        edges = helix_edges
        pk_adjacency: Dict[int, Set[int]] = {idx: set() for idx in pseudoknot_indices}
        for edge in edges:
            src = int(edge.src)
            dst = int(edge.dst)
            if src in pk_adjacency and dst in pk_adjacency:
                pk_adjacency[src].add(dst)
                pk_adjacency[dst].add(src)
        if not any(pk_adjacency[idx] for idx in pseudoknot_indices):
            return PseudoknotType.UNKNOWN
        remaining = set(pseudoknot_indices)
        stack = [next(iter(remaining))]
        seen: Set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(pk_adjacency[node] - seen)
        if seen != pseudoknot_indices:
            return PseudoknotType.COMPLEX
        if all(len(pk_adjacency[idx]) == 2 for idx in pseudoknot_indices):
            return PseudoknotType.M_TYPE if len(pseudoknot_indices) == 3 else PseudoknotType.COMPLEX
        return PseudoknotType.COMPLEX

    @staticmethod
    def _pseudoknot_span_from_stems(stems: Set[StemSegment | HelixSegment]) -> Pair | None:
        if not stems:
            return None
        start = min(stem.start_5p for stem in stems)
        stop = max(stem.start_3p for stem in stems)
        return start, stop

    @classmethod
    def _kissing_hairpin_signature(
        cls,
        pairs: List[Pair],
        length: int,
        *,
        nested_pairs: Tensor | np.ndarray | Pairs | None = None,
        pseudoknot_pairs: Tensor | np.ndarray | Pairs | None = None,
    ) -> Tuple[Loops | None, Tuple[int, int], int] | None:
        if nested_pairs is None or pseudoknot_pairs is None:
            nested_pairs, pseudoknot_pairs = split_pseudoknot_pairs(pairs)

        if isinstance(pseudoknot_pairs, Tensor):
            if pseudoknot_pairs.numel() == 0:
                return None
        elif isinstance(pseudoknot_pairs, np.ndarray):
            if pseudoknot_pairs.size == 0:
                return None
        else:
            if not pseudoknot_pairs:
                return None
        pk_segments = pairs_to_stem_segment_arrays(pseudoknot_pairs)
        start_i, start_j, lengths = pk_segments
        if isinstance(start_i, Tensor):
            count = int(start_i.numel())
        else:
            count = int(start_i.size) if hasattr(start_i, "size") else len(start_i)
        segments: List[Tuple[int, int, int]] = []
        for idx in range(count):
            seg_len = int(lengths[idx])
            if seg_len <= 0:
                continue
            segments.append((int(start_i[idx]), int(start_j[idx]), seg_len))
        if not segments:
            return None
        segments.sort(key=lambda segment: (segment[0], segment[1]))
        helix_count = 0
        current_len = segments[0][2]
        prev_start_i, prev_start_j, prev_len = segments[0]
        for next_start_i, next_start_j, next_len in segments[1:]:
            prev_end_i = prev_start_i + prev_len - 1
            prev_end_j = prev_start_j - prev_len + 1
            gap_i = next_start_i - prev_end_i - 1
            gap_j = prev_end_j - next_start_j - 1
            if 0 <= gap_i <= 1 and 0 <= gap_j <= 1:
                current_len += next_len
            else:
                if current_len >= 2:
                    helix_count += 1
                current_len = next_len
            prev_start_i, prev_start_j, prev_len = next_start_i, next_start_j, next_len
        if current_len >= 2:
            helix_count += 1
        if helix_count == 0:
            return None
        open_to_close = LoopSegments.open_to_close(nested_pairs, length)
        _roots, _children, intervals = compute_loop_spans(open_to_close, length)
        hairpin_intervals = [
            (start, end - 1) for start, end, code, _side in intervals if code == LoopSegmentType.HAIRPIN
        ]
        if not hairpin_intervals:
            return None
        hairpin_idx_by_pos = [-1] * length
        for idx, (start, end) in enumerate(hairpin_intervals):
            if start <= 0 or end + 1 >= length:
                return None
            close = int(open_to_close[start - 1].item())
            if close != end + 1:
                return None
            for pos in range(start, end + 1):
                hairpin_idx_by_pos[pos] = idx

        segment_loop_pairs: List[Tuple[int, int]] = []
        for start_5p, start_3p, seg_len in segments:
            loop_indices: Set[int] = set()
            for offset in range(seg_len):
                i = start_5p + offset
                j = start_3p - offset
                if i < 0 or j < 0 or i >= length or j >= length:
                    return None
                idx_i = hairpin_idx_by_pos[i]
                idx_j = hairpin_idx_by_pos[j]
                if idx_i == -1 or idx_j == -1 or idx_i == idx_j:
                    return None
                loop_indices.add(idx_i)
                loop_indices.add(idx_j)
            if len(loop_indices) != 2:
                return None
            sorted_indices = sorted(loop_indices)
            segment_loop_pairs.append((sorted_indices[0], sorted_indices[1]))
        if len(set(segment_loop_pairs)) != 1:
            return None
        idx1, idx2 = segment_loop_pairs[0]
        if helix_count != 1:
            return None, (idx1, idx2), helix_count

        nested_loops = cls._from_pairs_list(
            ensure_pairs_list(nested_pairs),
            length,
            has_crossing=False,
            pseudoknot_type=PseudoknotType.NONE,
            is_nested=True,
        )
        span_to_idx = {
            loop.spans[0]: idx
            for idx, loop in enumerate(nested_loops)
            if loop.kind == LoopType.HAIRPIN and len(loop.spans) == 1
        }
        span1 = LoopSpan(*hairpin_intervals[idx1])
        span2 = LoopSpan(*hairpin_intervals[idx2])
        loop_idx1 = span_to_idx.get(span1)
        loop_idx2 = span_to_idx.get(span2)
        if loop_idx1 is None or loop_idx2 is None:
            return None
        return nested_loops, (loop_idx1, loop_idx2), helix_count

    @classmethod
    def _kissing_hairpin_loops(
        cls,
        pairs: List[Pair],
        length: int,
        *,
        kissing_signature: Tuple[Loops | None, Tuple[int, int], int] | None = None,
    ) -> Loops | None:
        if kissing_signature is None:
            kissing_signature = cls._kissing_hairpin_signature(pairs, length)
        if kissing_signature is None:
            return None
        nested_loops, loop_pair, helix_count = kissing_signature
        if helix_count != 1 or nested_loops is None:
            return None
        idx1, idx2 = sorted(loop_pair, key=lambda idx: nested_loops[idx].spans[0].start)
        loops: List[Loop] = []
        for idx, loop in enumerate(nested_loops):
            if idx == idx1:
                loops.append(loop.with_taxonomy(PseudoknotType.KISSING_HAIRPIN, True, role=LoopRole.K1))
            elif idx == idx2:
                loops.append(loop.with_taxonomy(PseudoknotType.KISSING_HAIRPIN, True, role=LoopRole.K2))
            else:
                loops.append(loop)
        return cls(loops, length)


class LoopSegments:
    tier: int
    _intervals_all: Tuple[Tuple[int, int, LoopSegmentType, EndSide | None], ...]
    _intervals_by_kind: Dict[LoopSegmentType, List[Tuple[int, int, EndSide | None]]]
    _positions: Tensor

    def __init__(
        self,
        pairs: Tensor | np.ndarray | Pairs,
        length: int,
        device: torch.device | None = None,
        tier: int = 0,
    ):
        self.tier = tier
        pairs_list = ensure_pairs_list(normalize_pairs(pairs))
        pairs_device = pairs.device if isinstance(pairs, Tensor) else None
        if device is None:
            device = pairs_device
        if pairs_list and crossing_pairs(pairs_list):
            raise ValueError("LoopSegments does not support crossing pairs")
        self.open_to_close_map = LoopSegments.open_to_close(pairs, length)
        if device is not None and self.open_to_close_map.device != device:
            self.open_to_close_map = self.open_to_close_map.to(device)
        open_to_close_tensor = self.open_to_close_map.to(dtype=torch.long).view(-1)
        length = open_to_close_tensor.numel()
        if length == 0:
            self._intervals_all = ()
            self._intervals_by_kind = {
                LoopSegmentType.HAIRPIN: [],
                LoopSegmentType.BULGE: [],
                LoopSegmentType.INTERNAL: [],
                LoopSegmentType.BRANCH: [],
                LoopSegmentType.EXTERNAL: [],
                LoopSegmentType.END: [],
            }
            self._positions = torch.arange(0, dtype=torch.long, device=device)
            return

        roots, children, intervals = compute_loop_spans(open_to_close_tensor, length)

        intervals_by_kind: Dict[LoopSegmentType, List[Tuple[int, int, EndSide | None]]] = {
            LoopSegmentType.HAIRPIN: [],
            LoopSegmentType.BULGE: [],
            LoopSegmentType.INTERNAL: [],
            LoopSegmentType.BRANCH: [],
            LoopSegmentType.EXTERNAL: [],
            LoopSegmentType.END: [],
        }
        for start, end, code, side in intervals:
            intervals_by_kind[code].append((start, end, side))
        self._intervals_all = tuple(intervals)
        self._intervals_by_kind = intervals_by_kind
        self._positions = torch.arange(length, dtype=torch.long, device=device)

    def __repr__(self) -> str:
        length = int(self.open_to_close_map.numel())
        segment_count = len(self._intervals_all)
        parts = [f"tier={self.tier}", f"length={length}", f"segments={segment_count}"]
        parts.append(f"hairpin={len(self._intervals_by_kind[LoopSegmentType.HAIRPIN])}")
        parts.append(f"bulge={len(self._intervals_by_kind[LoopSegmentType.BULGE])}")
        parts.append(f"internal={len(self._intervals_by_kind[LoopSegmentType.INTERNAL])}")
        parts.append(f"branch={len(self._intervals_by_kind[LoopSegmentType.BRANCH])}")
        parts.append(f"external={len(self._intervals_by_kind[LoopSegmentType.EXTERNAL])}")
        parts.append(f"end={len(self._intervals_by_kind[LoopSegmentType.END])}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    @staticmethod
    def _build_tree(open_to_close: List[int], length: int) -> Tuple[List[int], List[List[int]]]:
        children: List[List[int]] = [[] for _ in range(length)]
        roots: List[int] = []
        stack: List[int] = []
        openers = [idx for idx, close in enumerate(open_to_close) if close != -1]
        closers = sorted(open_to_close[idx] for idx in openers) if openers else []
        open_idx = 0
        close_idx = 0
        while open_idx < len(openers) or close_idx < len(closers):
            next_open = openers[open_idx] if open_idx < len(openers) else length + 1
            next_close = closers[close_idx] if close_idx < len(closers) else length + 1
            if next_open < next_close:
                stack.append(next_open)
                open_idx += 1
                continue
            if stack and open_to_close[stack[-1]] == next_close:
                opener = stack.pop()
                parent = stack[-1] if stack else -1
                if parent == -1:
                    roots.append(opener)
                else:
                    children[parent].append(opener)
            close_idx += 1
        return roots, children

    @staticmethod
    def _build_intervals(
        open_to_close: List[int],
        roots: List[int],
        children: List[List[int]],
        length: int,
    ) -> List[Tuple[int, int, LoopSegmentType, EndSide | None]]:
        intervals: List[Tuple[int, int, LoopSegmentType, EndSide | None]] = []

        def mark(start: int, end: int, code: LoopSegmentType, side: EndSide | None = None) -> None:
            if start < end:
                intervals.append((start, end, code, side))

        prev_end = -1
        has_roots = bool(roots)
        for opener in roots:
            if prev_end + 1 < opener:
                if prev_end == -1:
                    mark(prev_end + 1, opener, LoopSegmentType.END, EndSide.FIVE_PRIME)
                else:
                    mark(prev_end + 1, opener, LoopSegmentType.EXTERNAL)
            prev_end = open_to_close[opener]

        if prev_end + 1 < length:
            side = None if not has_roots else EndSide.THREE_PRIME
            mark(prev_end + 1, length, LoopSegmentType.END, side)

        stack = roots[::-1]
        while stack:
            opener = stack.pop()
            closer = open_to_close[opener]
            child_openers = children[opener]
            if not child_openers:
                mark(opener + 1, closer, LoopSegmentType.HAIRPIN)
                continue

            if len(child_openers) == 1:
                child = child_openers[0]
                child_close = open_to_close[child]
                seg1 = (opener + 1, child) if opener + 1 < child else None
                seg2 = (child_close + 1, closer) if child_close + 1 < closer else None
                if seg1 and seg2:
                    mark(seg1[0], seg1[1], LoopSegmentType.INTERNAL)
                    mark(seg2[0], seg2[1], LoopSegmentType.INTERNAL)
                elif seg1:
                    mark(seg1[0], seg1[1], LoopSegmentType.BULGE)
                elif seg2:
                    mark(seg2[0], seg2[1], LoopSegmentType.BULGE)
            else:
                prev = opener
                for child in child_openers:
                    child_close = open_to_close[child]
                    if prev + 1 < child:
                        mark(prev + 1, child, LoopSegmentType.BRANCH)
                    prev = child_close
                if prev + 1 < closer:
                    mark(prev + 1, closer, LoopSegmentType.BRANCH)

            stack.extend(reversed(child_openers))

        intervals.sort(key=lambda seg: seg[0])
        if not intervals:
            paired = [False] * length
            for idx, close in enumerate(open_to_close):
                if close != -1:
                    paired[idx] = True
                    if 0 <= close < length:
                        paired[close] = True
            if all(paired):
                return []
            raise ValueError("no loop intervals could be constructed")
        return intervals

    @cached_property
    def _segments_cache(self) -> Tuple[LoopSegment, ...]:
        out_segments: List[LoopSegment] = []
        for start, end_exclusive, kind, side in self._intervals_all:
            stop = end_exclusive - 1
            if stop < start:
                continue
            out_segments.append(LoopSegment(kind, start, stop, self.tier, side))
        return tuple(out_segments)

    @cached_property
    def _interval_bounds(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if not self._intervals_all:
            return (), ()
        starts = tuple(start for start, _, _, _ in self._intervals_all)
        end_exclusive = tuple(end for _, end, _, _ in self._intervals_all)
        return starts, end_exclusive

    @cached_property
    def segments(self) -> List[LoopSegment]:
        return list(self._segments_cache)

    def segment_at(self, pos: int) -> LoopSegment | None:
        starts, end_exclusive = self._interval_bounds
        if not starts:
            return None
        idx = bisect_right(starts, pos) - 1
        if idx >= 0 and pos < end_exclusive[idx]:
            return self._segments_cache[idx]
        return None

    def positions(self, kind: LoopSegmentType | None = None, *, side: EndSide | None = None) -> List[Tensor]:
        if kind is None:
            intervals_all = self._intervals_all
            if side is None:
                return [self._positions[start:end] for start, end, _, _ in intervals_all]
            return [self._positions[start:end] for start, end, _, seg_side in intervals_all if seg_side == side]
        intervals_by_kind = self._intervals_by_kind[kind]
        if side is None:
            return [self._positions[start:end] for start, end, _ in intervals_by_kind]
        return [self._positions[start:end] for start, end, seg_side in intervals_by_kind if seg_side == side]

    @cached_property
    def hairpin_loops(self) -> List[LoopSegment]:
        return [segment for segment in self.segments if segment.kind == LoopSegmentType.HAIRPIN]

    @cached_property
    def bulge_segments(self) -> List[LoopSegment]:
        return [segment for segment in self.segments if segment.kind == LoopSegmentType.BULGE]

    @cached_property
    def internal_segments(self) -> List[LoopSegment]:
        return [segment for segment in self.segments if segment.kind == LoopSegmentType.INTERNAL]

    @cached_property
    def branch_segments(self) -> List[LoopSegment]:
        return [segment for segment in self.segments if segment.kind == LoopSegmentType.BRANCH]

    @cached_property
    def external_segments(self) -> List[LoopSegment]:
        return [segment for segment in self.segments if segment.kind == LoopSegmentType.EXTERNAL]

    @cached_property
    def end_5p(self) -> LoopSegment | None:
        segments = [segment for segment in self.segments if segment.kind == LoopSegmentType.END]
        for segment in segments:
            if segment.side == EndSide.FIVE_PRIME:
                return segment
        for segment in segments:
            if segment.side is None:
                return segment
        return None

    @cached_property
    def end_3p(self) -> LoopSegment | None:
        segments = [segment for segment in self.segments if segment.kind == LoopSegmentType.END]
        for segment in segments:
            if segment.side == EndSide.THREE_PRIME:
                return segment
        for segment in segments:
            if segment.side is None:
                return segment
        return None

    def contexts(self, pseudoknot_pairs: Tensor) -> List[LoopSegmentContext]:
        segments = self.segments
        pairs = pseudoknot_pairs
        if pairs.numel() == 0:
            return [LoopSegmentContext(segment, pairs, pairs) for segment in segments]
        if not segments:
            return []
        starts = torch.tensor([segment.start for segment in segments], device=pairs.device, dtype=pairs.dtype).view(
            -1, 1
        )
        stops = torch.tensor([segment.stop for segment in segments], device=pairs.device, dtype=pairs.dtype).view(-1, 1)
        left = pairs[:, 0].view(1, -1)
        right = pairs[:, 1].view(1, -1)
        in_left = (left >= starts) & (left <= stops)
        in_right = (right >= starts) & (right <= stops)
        inside_masks = in_left & in_right
        crossing_masks = in_left ^ in_right
        out: List[LoopSegmentContext] = []
        for idx, segment in enumerate(segments):
            inside = pairs[inside_masks[idx]]
            crossing = pairs[crossing_masks[idx]]
            out.append(LoopSegmentContext(segment, inside, crossing))
        return out

    @staticmethod
    def open_to_close(pairs: Tensor | np.ndarray | Pairs, length: int) -> Tensor:
        pairs = normalize_pairs(pairs)
        if isinstance(pairs, Tensor):
            if pairs.numel() == 0:
                return pairs.new_full((length,), -1, dtype=torch.long)
            pairs_tensor = pairs.to(dtype=torch.long)
        elif isinstance(pairs, np.ndarray):
            if pairs.size == 0:
                return torch.full((length,), -1, dtype=torch.long)
            pairs_tensor = torch.as_tensor(pairs, dtype=torch.long)
        else:
            if not pairs:
                return torch.full((length,), -1, dtype=torch.long)
            pairs_tensor = torch.as_tensor(pairs, dtype=torch.long)
        open_to_close = pairs_tensor.new_full((length,), -1, dtype=torch.long)
        left = torch.minimum(pairs_tensor[:, 0], pairs_tensor[:, 1])
        right = torch.maximum(pairs_tensor[:, 0], pairs_tensor[:, 1])
        open_to_close[left] = right
        return open_to_close


class DuplexSegmentsBase(Generic[SegmentT]):
    pairs: Tensor
    tier: int
    edges: List[StemEdge]
    graph: DirectedGraph

    def __init__(self, pairs: Tensor, tier: int):
        self.pairs = pairs
        self.tier = tier
        self.graph, self.edges = self._build_stem_graph()

    @classmethod
    def from_pairs(
        cls,
        pairs: Tensor | np.ndarray | Pairs,
        *,
        tier: int = 0,
        device: torch.device | None = None,
    ) -> DuplexSegmentsBase[SegmentT]:
        pairs = normalize_pairs(pairs)
        if isinstance(pairs, Tensor):
            if pairs.numel() == 0:
                if device is None:
                    empty = pairs.new_empty((0, 2), dtype=torch.long)
                else:
                    empty = torch.empty((0, 2), dtype=torch.long, device=device)
                return cls(empty, tier)
            if device is None:
                device = pairs.device
            return cls(pairs.to(device=device, dtype=torch.long), tier)
        if not pairs:
            empty = torch.empty((0, 2), dtype=torch.long, device=device)
            return cls(empty, tier)
        if isinstance(pairs, np.ndarray):
            pairs = torch.as_tensor(pairs, dtype=torch.long, device=device)
        elif isinstance(pairs, Sequence):
            pairs = torch.tensor(pairs, dtype=torch.long, device=device)
        return cls(pairs, tier)

    @classmethod
    def from_segment_list(
        cls,
        segments: Sequence[Segment],
        *,
        tier: int = 0,
        device: torch.device | None = None,
    ) -> DuplexSegmentsBase[SegmentT]:
        pairs_list: List[Pair] = []
        for segment in segments:
            for i, j in segment:
                pairs_list.append((int(i), int(j)))
        if pairs_list:
            pairs_list = normalize_pairs(pairs_list)
            pairs_tensor = torch.tensor(pairs_list, dtype=torch.long, device=device)
        else:
            pairs_tensor = torch.empty((0, 2), dtype=torch.long, device=device)
        instance = cls.__new__(cls)
        instance.pairs = pairs_tensor
        instance.tier = tier
        segment_objs = cls._from_segment_list(segments, tier)
        instance.__dict__["segments"] = segment_objs
        paired_positions = cls._paired_positions_from_pairs(pairs_tensor)
        instance.graph, instance.edges = cls._build_stem_graph_from_segments(
            segment_objs,
            paired_positions,
            tier=tier,
            device=pairs_tensor.device,
            stop_endpoint="5p",
            stop_target="3p",
        )
        return instance

    def __repr__(self) -> str:
        pair_count = int(self.pairs.shape[0])
        segments = self.__dict__.get("segments")
        segment_repr = "segments=?" if segments is None else f"segments={len(segments)}"
        parts = [
            f"tier={self.tier}",
            f"pairs={pair_count}",
            segment_repr,
            f"edges={len(self.edges)}",
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _build_stem_graph(self) -> Tuple[DirectedGraph, List[StemEdge]]:
        segments = self.segments
        paired_positions = self._paired_positions_from_pairs(self.pairs)
        return self._build_stem_graph_from_segments(
            segments,
            paired_positions,
            tier=self.tier,
            device=self.pairs.device,
            stop_endpoint="5p",
            stop_target="3p",
        )

    @classmethod
    def segments_by_tier(
        cls,
        pairs: Tensor | np.ndarray | Pairs,
        *,
        tiers: Sequence[int] | None = None,
        device: torch.device | None = None,
    ) -> List[SegmentT]:
        pairs = normalize_pairs(pairs)
        pairs_list = ensure_pairs_list(pairs)
        if not pairs_list:
            return []
        pairs_device = pairs.device if isinstance(pairs, Tensor) else None
        if device is None:
            device = pairs_device
        if isinstance(pairs, Tensor) and (device is None or pairs.device == device):
            pairs_tensor = pairs.to(dtype=torch.long)
            if device is None:
                device = pairs_tensor.device
        else:
            pairs_tensor = torch.tensor(pairs_list, dtype=torch.long, device=device)
        tier_pairs = cls._tier_pairs(pairs_tensor)
        tier_indices: Sequence[int]
        if tiers is None:
            tier_indices = range(len(tier_pairs))
        else:
            for tier_idx in tiers:
                if tier_idx < 0 or tier_idx >= len(tier_pairs):
                    raise IndexError("tier is out of range")
            tier_indices = tuple(tiers)
        segments: List[SegmentT] = []
        for tier_idx in tier_indices:
            segments.extend(cls._from_pairs(tier_pairs[tier_idx], tier_idx))
        segments.sort(key=cls._segment_sort_key)
        return segments

    @cached_property
    def segments(self) -> List[SegmentT]:
        return self._from_pairs(self.pairs, self.tier)

    @cached_property
    def pair_map(self) -> Dict[Pair, SegmentT]:
        return self._pair_map_for_segments(self.segments)

    @staticmethod
    def _segment_sort_key(segment: StemSegment | HelixSegment) -> Tuple[int, int, int, int, int]:
        return (segment.tier, segment.start_5p, segment.start_3p, segment.stop_5p, segment.stop_3p)

    @staticmethod
    def _from_pairs(pairs: Tensor, tier: int) -> List[SegmentT]:
        raise NotImplementedError

    @staticmethod
    def _from_segment_list(segments: Sequence[Segment], tier: int) -> List[SegmentT]:
        raise NotImplementedError

    @staticmethod
    def _pair_map_for_segments(segments: List[SegmentT]) -> Dict[Pair, SegmentT]:
        raise NotImplementedError

    @staticmethod
    def _paired_positions_from_pairs(pairs: Tensor) -> List[int]:
        if pairs.numel() == 0:
            return []
        flat = pairs.detach().to(device="cpu", dtype=torch.long).view(-1)
        positions = torch.sort(torch.unique(flat)).values
        return positions.tolist()

    @staticmethod
    def _tier_pairs(pairs: Tensor) -> List[Tensor]:
        if pairs.numel() == 0:
            return [pairs.new_empty((0, 2), dtype=torch.long)]
        nested, pseudoknot = split_pseudoknot_pairs(pairs)
        empty = pairs.new_empty((0, 2), dtype=torch.long)
        tiers: List[Tensor] = [nested if nested.numel() else empty]
        if pseudoknot.numel() == 0:
            return tiers
        pk_tiers = pseudoknot_tiers(pseudoknot)
        if not pk_tiers:
            tiers.append(empty)
            return tiers
        for tier in pk_tiers:
            if isinstance(tier, Tensor):
                tiers.append(tier)
            else:
                tiers.append(torch.tensor(tier, dtype=torch.long, device=pairs.device))
        return tiers

    @staticmethod
    def _build_stem_graph_from_segments(
        segments: Sequence[StemSegment | HelixSegment],
        paired_positions: List[int],
        *,
        tier: int,
        device: torch.device,
        stop_endpoint: str = "5p",
        stop_target: str = "3p",
    ) -> Tuple[DirectedGraph, List[StemEdge]]:
        graph = DirectedGraph(device=device, allow_multi_edges=True)
        edges: List[StemEdge] = []
        if not segments:
            return graph, edges
        if stop_endpoint not in ("3p", "5p") or stop_target not in ("3p", "5p"):
            raise ValueError("stop_endpoint and stop_target must be '3p' or '5p'")
        start_pos_to_segments: Dict[int, List[int]] = {}
        stop_pos_to_segments: Dict[int, List[int]] = {}
        for idx, segment in enumerate(segments):
            start_pos_to_segments.setdefault(segment.start_5p, []).append(idx)
            stop_pos = segment.stop_5p if stop_target == "5p" else segment.stop_3p
            stop_pos_to_segments.setdefault(stop_pos, []).append(idx)

        endpoints = [segment.start_3p for segment in segments]
        if stop_endpoint == "3p":
            endpoints.extend(segment.stop_3p for segment in segments)
        else:
            endpoints.extend(segment.stop_5p for segment in segments)
        next_pos_by_endpoint = DuplexSegmentsBase._next_positions_map(paired_positions, endpoints)

        for idx, segment in enumerate(segments):
            graph.add_node(idx)
            next_start = next_pos_by_endpoint.get(segment.start_3p)
            if next_start is not None:
                for dst in start_pos_to_segments.get(next_start, []):
                    if dst == idx:
                        continue
                    graph.add_edge(idx, dst, attr={"type": StemEdgeType.START_START.value})
                    edges.append(StemEdge(idx, dst, segment.start_3p, next_start, StemEdgeType.START_START, tier))
                for dst in stop_pos_to_segments.get(next_start, []):
                    if dst == idx:
                        continue
                    graph.add_edge(idx, dst, attr={"type": StemEdgeType.START_STOP.value})
                    edges.append(StemEdge(idx, dst, segment.start_3p, next_start, StemEdgeType.START_STOP, tier))

            stop_pos = segment.stop_3p if stop_endpoint == "3p" else segment.stop_5p
            next_stop = next_pos_by_endpoint.get(stop_pos)
            if next_stop is not None:
                for dst in start_pos_to_segments.get(next_stop, []):
                    if dst == idx:
                        continue
                    graph.add_edge(idx, dst, attr={"type": StemEdgeType.STOP_START.value})
                    edges.append(StemEdge(idx, dst, stop_pos, next_stop, StemEdgeType.STOP_START, tier))
                for dst in stop_pos_to_segments.get(next_stop, []):
                    if dst == idx:
                        continue
                    graph.add_edge(idx, dst, attr={"type": StemEdgeType.STOP_STOP.value})
                    edges.append(StemEdge(idx, dst, stop_pos, next_stop, StemEdgeType.STOP_STOP, tier))
        return graph, edges

    @staticmethod
    def _next_positions_map(positions: List[int], endpoints: Sequence[int]) -> Dict[int, int | None]:
        if not endpoints:
            return {}
        if not positions:
            return dict.fromkeys(sorted(endpoints), None)
        return {pos: DuplexSegmentsBase._next_paired_position(positions, pos) for pos in set(endpoints)}

    @staticmethod
    def _next_paired_position(positions: List[int], pos: int) -> int | None:
        if not positions:
            return None
        idx = bisect_right(positions, pos)
        if idx >= len(positions):
            return None
        return positions[idx]


class HelixSegments(DuplexSegmentsBase[HelixSegment]):

    pairs: Tensor
    tier: int
    edges: List[StemEdge]
    graph: DirectedGraph

    @staticmethod
    def _pair_map_for_segments(segments: List[HelixSegment]) -> Dict[Pair, HelixSegment]:
        mapping: Dict[Pair, HelixSegment] = {}
        for segment in segments:
            for i, j in segment.pairs:
                if i > j:
                    i, j = j, i
                mapping[(int(i), int(j))] = segment
        return mapping

    @staticmethod
    def _from_pairs(pairs: Tensor, tier: int) -> List[HelixSegment]:
        if pairs.numel() == 0:
            return []
        out_segments: List[HelixSegment] = []
        start_i, start_j, lengths = pairs_to_helix_segment_arrays(pairs)
        count = int(start_i.numel()) if isinstance(start_i, Tensor) else int(start_i.size)
        for idx in range(count):
            seg_len = int(lengths[idx])
            if seg_len <= 0:
                continue
            start_5p = int(start_i[idx])
            start_3p = int(start_j[idx])
            stop_5p = start_5p + seg_len - 1
            stop_3p = start_3p - seg_len + 1
            pairs_list = tuple((start_5p + offset, start_3p - offset) for offset in range(seg_len))
            out_segments.append(
                HelixSegment(
                    start_5p,
                    stop_5p,
                    start_3p,
                    stop_3p,
                    pairs_list,
                    tier,
                )
            )
        out_segments.sort(key=lambda segment: (segment.start_5p, segment.start_3p, segment.stop_5p, segment.stop_3p))
        return out_segments

    @staticmethod
    def _from_segment_list(segments: Sequence[Segment], tier: int) -> List[HelixSegment]:
        out_segments: List[HelixSegment] = []
        for segment in segments:
            if not segment:
                continue
            first = segment[0]
            last = segment[-1] if len(segment) > 1 else first
            start_5p, start_3p = first
            stop_5p, stop_3p = last
            pairs_list = tuple((int(i), int(j)) for i, j in segment)
            out_segments.append(
                HelixSegment(
                    int(start_5p),
                    int(stop_5p),
                    int(start_3p),
                    int(stop_3p),
                    pairs_list,
                    tier,
                )
            )
        out_segments.sort(key=lambda segment: (segment.start_5p, segment.start_3p, segment.stop_5p, segment.stop_3p))
        return out_segments


class StemSegments(DuplexSegmentsBase[StemSegment]):

    pairs: Tensor
    tier: int
    edges: List[StemEdge]
    graph: DirectedGraph

    @classmethod
    def pair_map_by_tiers(cls, pairs: List[Pair]) -> Dict[Pair, StemSegment]:
        if not pairs:
            return {}
        pairs_tensor = torch.tensor(pairs, dtype=torch.long)
        tiers = cls._tier_pairs(pairs_tensor)
        mapping: Dict[Pair, StemSegment] = {}
        for tier_idx, tier_pairs in enumerate(tiers):
            segments = cls._from_pairs(tier_pairs, tier_idx)
            mapping.update(cls._pair_map_for_segments(segments))
        return mapping

    @staticmethod
    def _pair_map_for_segments(segments: List[StemSegment]) -> Dict[Pair, StemSegment]:
        mapping: Dict[Pair, StemSegment] = {}
        for segment in segments:
            seg_len = len(segment)
            for offset in range(seg_len):
                i = segment.start_5p + offset
                j = segment.start_3p - offset
                if i > j:
                    i, j = j, i
                mapping[(i, j)] = segment
        return mapping

    def adjacency(self) -> Dict[int, Set[int]]:
        stems = self.segments
        if not stems:
            return {}
        paired_positions = self._paired_positions()
        if not paired_positions:
            return {idx: set() for idx in range(len(stems))}
        endpoint_to_stems: Dict[int, List[int]] = {}
        endpoints: List[int] = []
        for idx, stem in enumerate(stems):
            for pos in (stem.start_5p, stem.stop_5p, stem.start_3p, stem.stop_3p):
                endpoint_to_stems.setdefault(pos, []).append(idx)
                endpoints.append(pos)
        next_pos_by_endpoint = self._next_positions_map(paired_positions, endpoints)
        adjacency: Dict[int, Set[int]] = {idx: set() for idx in range(len(stems))}
        for idx, stem in enumerate(stems):
            for pos in (stem.start_5p, stem.stop_5p, stem.start_3p, stem.stop_3p):
                next_pos = next_pos_by_endpoint.get(pos)
                if next_pos is None:
                    continue
                for dst in endpoint_to_stems.get(next_pos, []):
                    if dst == idx:
                        continue
                    adjacency[idx].add(dst)
                    adjacency[dst].add(idx)
        return adjacency

    @staticmethod
    def _from_pairs(pairs: Tensor, tier: int) -> List[StemSegment]:
        if pairs.numel() == 0:
            return []
        start_i, start_j, lengths = pairs_to_stem_segment_arrays(pairs)
        out_segments = stem_segment_arrays_to_stem_segment_list(start_i, start_j, lengths, tier=tier)
        out_segments.sort(key=lambda segment: (segment.start_5p, segment.start_3p, segment.stop_5p, segment.stop_3p))
        return out_segments

    @staticmethod
    def _from_segment_list(segments: Sequence[Segment], tier: int) -> List[StemSegment]:
        out_segments: List[StemSegment] = []
        for segment in segments:
            if not segment:
                continue
            first = segment[0]
            last = segment[-1] if len(segment) > 1 else first
            start_5p, start_3p = first
            stop_5p, stop_3p = last
            out_segments.append(StemSegment(int(start_5p), int(stop_5p), int(start_3p), int(stop_3p), tier))
        out_segments.sort(key=lambda segment: (segment.start_5p, segment.start_3p, segment.stop_5p, segment.stop_3p))
        return out_segments

    def _paired_positions(self) -> List[int]:
        return self._paired_positions_from_pairs(self.pairs)
