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

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from ...graph import UndirectedGraph
from .bprna import annotate_function, annotate_structure
from .notations import _DOT_BRACKET_PAIR_TABLE, _REVERSE_DOT_BRACKET_PAIR_TABLE, _UNPAIRED_TOKENS
from .pseudoknot import _torch_crossing_mask


class EdgeType(IntEnum):
    BACKBONE = 0
    PRIMARY_PAIRS = 1
    PSEUDOKNOT_PAIR = 2


class LoopType(IntEnum):
    HAIRPIN = 1
    BULGE = 2
    INTERNAL = 3
    BRANCH = 4
    EXTERNAL = 5
    END = 6


@dataclass
class Loops:
    hairpins: List[Tensor]
    bulges: List[Tensor]
    internals: List[Tensor]
    branches: List[Tensor]
    externals: List[Tensor]
    ends: List[Tensor]
    loops: List[Tensor]


class RnaSecondaryStructure(UndirectedGraph):
    """RNA secondary structure graph with metadata (sequence + dot-bracket)."""

    def __init__(self, sequence: str, secondary_structure: str, device: torch.device | None = None, **kwargs):
        if len(sequence) != len(secondary_structure):
            raise ValueError("sequence and secondary_structure must have the same length")

        # Parse and split pairs once
        pairs_np, primary_pairs_np, pseudoknot_pairs_np, has_pseudoknot, primary_open_to_close = _read_dot_brackets(
            secondary_structure
        )
        all_pairs = torch.from_numpy(pairs_np)
        if device is not None:
            all_pairs = all_pairs.to(device=device)
        if has_pseudoknot:
            primary_pairs = torch.from_numpy(primary_pairs_np)
            pseudoknot_pairs = torch.from_numpy(pseudoknot_pairs_np)
            if device is not None:
                primary_pairs = primary_pairs.to(device=device)
                pseudoknot_pairs = pseudoknot_pairs.to(device=device)
        else:
            primary_pairs = all_pairs
            pseudoknot_pairs = all_pairs.new_empty((0, 2))

        length = len(secondary_structure)
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
        if len(primary_pairs):
            edge_index_parts.append(primary_pairs)
            edge_type_parts.append(
                torch.full(
                    (len(primary_pairs), 1),
                    EdgeType.PRIMARY_PAIRS.value,
                    dtype=torch.long,
                    device=primary_pairs.device,
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
        self._all_pairs = all_pairs
        self._primary_pairs = primary_pairs
        self._pseudoknot_pairs = pseudoknot_pairs

        self._stems_ptr = None
        self._primary_stems_ptr = None
        self._pseudoknot_stems_ptr = None
        self._stems = None
        self._primary_stems = None
        self._pseudoknot_stems = None

        self._loop_labels, self._loop_segments = _build_loop_labels_and_segments(
            primary_open_to_close, length, self._primary_pairs.device
        )
        self._loop_tensors = None

        super().__init__(edge_index=edge_index, edge_features={"type": edge_types}, device=device, **kwargs)

    @property
    def pairs(self) -> Tensor:
        return self._all_pairs

    @property
    def primary_pairs(self) -> Tensor:
        return self._primary_pairs

    @property
    def pseudoknot_pairs(self) -> Tensor:
        return self._pseudoknot_pairs

    @property
    def crossing_pairs(self) -> Tensor:
        """Return pairs that participate in any crossing (pseudoknot event)."""
        if len(self._all_pairs) < 2:
            return self._all_pairs
        mask = _torch_crossing_mask(self._all_pairs)
        return self._all_pairs[mask]

    @property
    def stems(self):
        if self._stems is None:
            self._stems = _stems_from_ptr(self._all_pairs, self.stems_ptr)
        return self._stems

    @property
    def primary_stems(self):
        if self._primary_stems is None:
            self._primary_stems = _stems_from_ptr(self._primary_pairs, self.primary_stems_ptr)
        return self._primary_stems

    @property
    def pseudoknot_stems(self):
        if self._pseudoknot_stems is None:
            self._pseudoknot_stems = _stems_from_ptr(self._pseudoknot_pairs, self.pseudoknot_stems_ptr)
        return self._pseudoknot_stems

    @property
    def loops(self) -> Loops:
        if self._loop_tensors is None:
            self._loop_tensors = _segments_to_loops(  # type: ignore[assignment]
                self._loop_segments, len(self._loop_labels), self._loop_labels.device
            )
        return self._loop_tensors  # type: ignore[return-value]

    @property
    def hairpins(self):
        return self.loops.hairpins

    @property
    def bulges(self):
        return self.loops.bulges

    @property
    def internals(self):
        return self.loops.internals

    @property
    def branches(self):
        return self.loops.branches

    @property
    def externals(self):
        return self.loops.externals

    @property
    def ends(self):
        return self.loops.ends

    @property
    def loop_labels(self) -> Tensor:
        return self._loop_labels

    @property
    def stems_ptr(self) -> Tensor:
        if self._stems_ptr is None:
            self._stems_ptr = _find_stem_ptr(self._all_pairs)
        return self._stems_ptr

    @property
    def primary_stems_ptr(self) -> Tensor:
        if self._primary_stems_ptr is None:
            self._primary_stems_ptr = _find_stem_ptr(self._primary_pairs)
        return self._primary_stems_ptr

    @property
    def pseudoknot_stems_ptr(self) -> Tensor:
        if self._pseudoknot_stems_ptr is None:
            self._pseudoknot_stems_ptr = _find_stem_ptr(self._pseudoknot_pairs)
        return self._pseudoknot_stems_ptr

    @property
    def structural_annotation(self) -> str:
        return annotate_structure(self)

    @property
    def functional_annotation(self) -> str:
        return annotate_function(self)


def _find_stem_ptr(pairs: Tensor) -> Tensor:
    if len(pairs) == 0:
        return pairs.new_tensor([0], dtype=torch.long)
    if len(pairs) == 1:
        return pairs.new_tensor([0, 1], dtype=torch.long)
    cont = (pairs[1:, 0] == pairs[:-1, 0] + 1) & (pairs[1:, 1] == pairs[:-1, 1] - 1)
    breaks = torch.nonzero(~cont, as_tuple=False).flatten() + 1
    return torch.cat([pairs.new_tensor([0]), breaks, pairs.new_tensor([len(pairs)])])


def _stems_from_ptr(pairs: Tensor, stem_ptr: Tensor) -> List[Tensor]:
    if len(stem_ptr) <= 1:
        return []
    sizes = torch.diff(stem_ptr).to(dtype=torch.long).cpu().tolist()
    return list(torch.split(pairs, sizes, dim=0))


def _read_dot_brackets(dot_bracket: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    length = len(dot_bracket)
    paren_stack: List[int] = []
    other_stacks: Dict[str, List[int]] = {}
    primary_open_to_close = np.full(length, -1, dtype=np.int64)
    all_open_to_close: np.ndarray | None = None
    has_pseudoknot = False
    openers = _DOT_BRACKET_PAIR_TABLE
    closers = _REVERSE_DOT_BRACKET_PAIR_TABLE
    unpaired = _UNPAIRED_TOKENS
    for i, symbol in enumerate(dot_bracket):
        if symbol == "(":
            paren_stack.append(i)
            continue
        if symbol == ")":
            if not paren_stack:
                raise ValueError(f"Unmatched symbol {symbol} at position {i} in sequence {dot_bracket}")
            j = paren_stack.pop()
            primary_open_to_close[j] = i
            if all_open_to_close is not None:
                all_open_to_close[j] = i
            continue

        if symbol in unpaired:
            continue

        opener = openers.get(symbol)
        if opener is not None:
            if all_open_to_close is None:
                all_open_to_close = primary_open_to_close.copy()
            has_pseudoknot = True
            other_stacks.setdefault(symbol, []).append(i)
            continue

        opener = closers.get(symbol)
        if opener is not None:
            stack = other_stacks.get(opener)
            if not stack:
                raise ValueError(f"Unmatched symbol {symbol} at position {i} in sequence {dot_bracket}")
            if all_open_to_close is None:
                all_open_to_close = primary_open_to_close.copy()
            has_pseudoknot = True
            j = stack.pop()
            all_open_to_close[j] = i
            continue

        raise ValueError(f"Invalid symbol {symbol} at position {i} in sequence {dot_bracket}")

    if paren_stack:
        raise ValueError(f"Unmatched symbol ( at position {paren_stack[0]} in sequence {dot_bracket}")
    for symbol, stack in other_stacks.items():
        if stack:
            raise ValueError(f"Unmatched symbol {symbol} at position {stack[0]} in sequence {dot_bracket}")

    if all_open_to_close is None:
        openers_idx = np.flatnonzero(primary_open_to_close != -1)
        if openers_idx.size == 0:
            empty = np.empty((0, 2), dtype=int)
            return empty, empty, empty, has_pseudoknot, primary_open_to_close
        pairs_np = np.empty((openers_idx.size, 2), dtype=np.int64)
        pairs_np[:, 0] = openers_idx
        pairs_np[:, 1] = primary_open_to_close[openers_idx]
        empty = np.empty((0, 2), dtype=int)
        return pairs_np, pairs_np, empty, has_pseudoknot, primary_open_to_close

    openers_idx = np.flatnonzero(all_open_to_close != -1)
    if openers_idx.size == 0:
        empty = np.empty((0, 2), dtype=int)
        return empty, empty, empty, has_pseudoknot, primary_open_to_close
    pairs_np = np.empty((openers_idx.size, 2), dtype=np.int64)
    pairs_np[:, 0] = openers_idx
    pairs_np[:, 1] = all_open_to_close[openers_idx]
    paren_openers = np.flatnonzero(primary_open_to_close != -1)
    if paren_openers.size == 0:
        paren_pairs_np = np.empty((0, 2), dtype=int)
    else:
        paren_pairs_np = np.empty((paren_openers.size, 2), dtype=np.int64)
        paren_pairs_np[:, 0] = paren_openers
        paren_pairs_np[:, 1] = primary_open_to_close[paren_openers]
    pseudoknot_openers = np.flatnonzero((all_open_to_close != -1) & (primary_open_to_close == -1))
    if pseudoknot_openers.size == 0:
        pseudoknot_pairs_np = np.empty((0, 2), dtype=int)
    else:
        pseudoknot_pairs_np = np.empty((pseudoknot_openers.size, 2), dtype=np.int64)
        pseudoknot_pairs_np[:, 0] = pseudoknot_openers
        pseudoknot_pairs_np[:, 1] = all_open_to_close[pseudoknot_openers]
    return pairs_np, paren_pairs_np, pseudoknot_pairs_np, has_pseudoknot, primary_open_to_close


def _build_loop_labels_and_segments(
    open_to_close: np.ndarray, length: int, device: torch.device
) -> Tuple[Tensor, List[Tuple[int, int, int]]]:
    if length == 0:
        return torch.empty((0,), dtype=torch.int8, device=device), []

    has_open = open_to_close != -1
    if not np.any(has_open):
        labels_t = torch.full((length,), LoopType.END, dtype=torch.int8, device=device)
        return labels_t, [(0, length, LoopType.END)] if length else []

    children: List[List[int]] = [[] for _ in range(length)]
    roots: List[int] = []
    stack: List[int] = []
    for pos in range(length):
        if has_open[pos]:
            stack.append(pos)
        while stack and open_to_close[stack[-1]] == pos:
            opener = stack.pop()
            parent = stack[-1] if stack else -1
            if parent == -1:
                roots.append(opener)
            else:
                children[parent].append(opener)

    labels = np.zeros(length, dtype=np.int8)
    segments: List[Tuple[int, int, int]] = []

    def mark(start: int, end: int, code: int) -> None:
        if start < end:
            labels[start:end] = code
            if code:
                segments.append((start, end, code))

    prev_end = -1
    for opener in roots:
        if prev_end + 1 < opener:
            code = LoopType.END if prev_end == -1 else LoopType.EXTERNAL
            mark(prev_end + 1, opener, code)
        prev_end = open_to_close[opener]

    if prev_end + 1 < length:
        mark(prev_end + 1, length, LoopType.END)

    stack = roots[::-1]
    while stack:
        opener = stack.pop()
        closer = open_to_close[opener]
        child_openers = children[opener]
        if not child_openers:
            mark(opener + 1, closer, LoopType.HAIRPIN)
            continue

        if len(child_openers) == 1:
            child = child_openers[0]
            child_close = open_to_close[child]
            seg1 = (opener + 1, child) if opener + 1 < child else None
            seg2 = (child_close + 1, closer) if child_close + 1 < closer else None
            if seg1 and seg2:
                mark(seg1[0], seg1[1], LoopType.INTERNAL)
                mark(seg2[0], seg2[1], LoopType.INTERNAL)
            elif seg1:
                mark(seg1[0], seg1[1], LoopType.BULGE)
            elif seg2:
                mark(seg2[0], seg2[1], LoopType.BULGE)
        else:
            prev = opener
            for child in child_openers:
                child_close = open_to_close[child]
                if prev + 1 < child:
                    mark(prev + 1, child, LoopType.BRANCH)
                prev = child_close
            if prev + 1 < closer:
                mark(prev + 1, closer, LoopType.BRANCH)

        if child_openers:
            stack.extend(reversed(child_openers))

    labels_t = torch.from_numpy(labels)
    if device.type != "cpu":
        labels_t = labels_t.to(device=device)
    if segments:
        segments.sort(key=lambda seg: seg[0])
    return labels_t, segments


def _segments_to_loops(segments: List[Tuple[int, int, int]], length: int, device: torch.device) -> Loops:
    hairpins: List[Tensor] = []
    bulges: List[Tensor] = []
    internals: List[Tensor] = []
    branches: List[Tensor] = []
    externals: List[Tensor] = []
    ends: List[Tensor] = []
    loops: List[Tensor] = []

    if not segments:
        return Loops(
            hairpins=hairpins,
            bulges=bulges,
            internals=internals,
            branches=branches,
            externals=externals,
            ends=ends,
            loops=loops,
        )

    positions = torch.arange(length, dtype=torch.long, device=device)
    for start, end, code in segments:
        segment = positions[start:end]
        if code == LoopType.HAIRPIN:
            hairpins.append(segment)
        elif code == LoopType.BULGE:
            bulges.append(segment)
        elif code == LoopType.INTERNAL:
            internals.append(segment)
        elif code == LoopType.BRANCH:
            branches.append(segment)
        elif code == LoopType.EXTERNAL:
            externals.append(segment)
        elif code == LoopType.END:
            ends.append(segment)
        else:
            continue
        loops.append(segment)

    return Loops(
        hairpins=hairpins,
        bulges=bulges,
        internals=internals,
        branches=branches,
        externals=externals,
        ends=ends,
        loops=loops,
    )
