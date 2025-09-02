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

from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import torch

from .noncanonical import noncanonical_pairs_set
from .pairs import Pair, PairMap, Pairs, ensure_pairs_list
from .topology import HelixSegments, RnaSecondaryStructureTopology, compute_loop_spans
from .types import LoopSegmentType

STRUCTURE_TYPE_KEYS = ("S", "H", "B", "I", "M", "X", "E", "PK", "PKBP", "NCBP", "SEGMENTS")

_SEGMENT_TYPE_TO_CHAR: Dict[LoopSegmentType, str] = {
    LoopSegmentType.HAIRPIN: "H",
    LoopSegmentType.BULGE: "B",
    LoopSegmentType.INTERNAL: "I",
    LoopSegmentType.BRANCH: "M",
    LoopSegmentType.EXTERNAL: "X",
    LoopSegmentType.END: "E",
}


class BpRnaSecondaryStructureTopology:

    pair_map: PairMap
    topology: RnaSecondaryStructureTopology

    def __init__(
        self,
        sequence: str,
        pairs: torch.Tensor | np.ndarray | Pairs | None = None,
        dot_bracket: str | None = None,
        *,
        topology: RnaSecondaryStructureTopology | None = None,
    ):
        if topology is None:
            if dot_bracket is not None and len(dot_bracket) != len(sequence):
                raise ValueError("dot_bracket length must match sequence length")
            if pairs is None:
                if dot_bracket is None:
                    raise ValueError("pairs or dot_bracket must be provided")
                topology = RnaSecondaryStructureTopology(sequence, dot_bracket)
            else:
                device = pairs.device if isinstance(pairs, torch.Tensor) else torch.device("cpu")
                pairs_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
                topology = RnaSecondaryStructureTopology(sequence, pairs_tensor)
        else:
            if sequence != topology.sequence:
                raise ValueError("sequence must match topology.sequence")

        self.topology = topology
        self.sequence = topology.sequence
        nested_pair_map = PairMap(topology.nested_pairs)

        self.pair_map = nested_pair_map
        self._pseudoknot_pairs = ensure_pairs_list(topology.pseudoknot_pairs)

    def __len__(self) -> int:
        return len(self.sequence)

    @staticmethod
    def _safe_base(sequence: str, idx: int) -> str:
        if idx < 0 or idx >= len(sequence):
            return ""
        return sequence[idx]

    @staticmethod
    def _slice_sequence(sequence: str, start: int, stop: int) -> str:
        if start < 0 or stop < 0 or start > stop:
            return ""
        return sequence[start : stop + 1]

    @staticmethod
    def _format_range(start: int, stop: int) -> str:
        return f"{start + 1}..{stop + 1}"

    @staticmethod
    def _format_pos(pos: int) -> str:
        return str(pos + 1)

    @cached_property
    def structural_annotation(self) -> str:
        length = len(self.sequence)
        if length <= 0:
            return ""

        s_array = np.full(length, "E", dtype="<U1")
        pairs_arr = np.asarray(self.pair_map.pairs, dtype=int)
        if pairs_arr.size:
            flat = pairs_arr.reshape(-1)
            valid = (flat >= 0) & (flat < length)
            if valid.any():
                s_array[flat[valid]] = "S"
        loop_segments = self.topology.nested_loop_segments
        segments = loop_segments.segments
        if segments:
            positions = np.arange(length, dtype=int)
            starts = np.array([seg.start for seg in segments], dtype=int)
            stops = np.array([seg.stop for seg in segments], dtype=int)
            kinds = np.array([seg.kind for seg in segments])
            mask = (positions[None, :] >= starts[:, None]) & (positions[None, :] <= stops[:, None])
            for kind in np.unique(kinds):
                char = _SEGMENT_TYPE_TO_CHAR.get(kind)
                if char is None:
                    continue
                kind_mask = kinds == kind
                if not kind_mask.any():
                    continue
                seg_mask = mask[kind_mask].any(axis=0)
                update_mask = (s_array != "S") & seg_mask
                if update_mask.any():
                    s_array[update_mask] = char
        return "".join(s_array.tolist())

    @cached_property
    def functional_annotation(self) -> str:
        length = len(self.sequence)
        if length <= 0:
            return ""
        positions = set(self.topology.paired_positions(view="pseudoknot"))
        return "".join("K" if idx in positions else "N" for idx in range(length))

    @cached_property
    def structure_types(self) -> Dict[str, List[str]]:
        pk_pair_map = PairMap(self._pseudoknot_pairs)
        pseudoknot_segments = HelixSegments.from_segment_list(pk_pair_map.segments).segments
        pair_map = self.pair_map
        segment_pairs_list = self.pair_map.segments
        structure_types: Dict[str, List[str]] = {key: [] for key in STRUCTURE_TYPE_KEYS}
        pseudoknot_loop_segments: Dict[int, List[Tuple[str, int, int]]] = {}
        pseudoknot_pair_set = set(pk_pair_map.pairs)
        pseudoknot_positions = set(self.topology.paired_positions(view="pseudoknot"))
        loop_segments = self.topology.nested_loop_segments
        knot_bounds: List[Tuple[int, int, int, int]] = []
        if pseudoknot_segments:
            for knot in pseudoknot_segments:
                knot_pairs = list(knot.pairs)
                if not knot_pairs:
                    continue
                k_5p_start = knot.start_5p
                k_3p_start = knot.start_3p
                k_5p_stop = knot.stop_5p
                k_3p_stop = knot.stop_3p
                knot_bounds.append((k_5p_start, k_3p_start, k_5p_stop, k_3p_stop))

        knot_bounds_arr: np.ndarray | None = None
        if knot_bounds:
            knot_bounds_arr = np.asarray(knot_bounds, dtype=int)
            knot_5p_start = knot_bounds_arr[:, 0]
            knot_3p_start = knot_bounds_arr[:, 1]
            knot_5p_stop = knot_bounds_arr[:, 2]
            knot_3p_stop = knot_bounds_arr[:, 3]

        def knots_for_span(start: int, stop: int) -> List[int]:
            if start > stop or knot_bounds_arr is None:
                return []
            mask = ((start <= knot_5p_start) & (knot_5p_stop <= stop)) | (
                (start <= knot_3p_stop) & (knot_3p_start <= stop)
            )
            if not mask.any():
                return []
            return (np.nonzero(mask)[0] + 1).tolist()

        hairpin_spans: List[Tuple[int, int, List[int]]] = []
        bulge_loops: List[Tuple[int, int, List[int]]] = []
        multi_loops: List[List[Tuple[int, int, List[int]]]] = []
        end_loops: List[Tuple[int, int, List[int]]] = []
        external_spans: List[Tuple[int, int, List[int]]] = []
        internal_loops: List[List[Tuple[int, int, List[int]]]] = []
        internal_spans: set[Tuple[int, int]] = set()
        for loop_segment in loop_segments.segments:
            knots_in = knots_for_span(loop_segment.start, loop_segment.stop)
            if loop_segment.kind == LoopSegmentType.HAIRPIN:
                hairpin_spans.append((loop_segment.start, loop_segment.stop, knots_in))
            elif loop_segment.kind == LoopSegmentType.BULGE:
                bulge_loops.append((loop_segment.start, loop_segment.stop, knots_in))
            elif loop_segment.kind == LoopSegmentType.INTERNAL:
                internal_spans.add((loop_segment.start, loop_segment.stop))
            elif loop_segment.kind == LoopSegmentType.EXTERNAL:
                external_spans.append((loop_segment.start, loop_segment.stop, knots_in))
            elif loop_segment.kind == LoopSegmentType.END:
                end_loops.append((loop_segment.start, loop_segment.stop, knots_in))

        for i, j in pair_map.pairs:
            if i < j and j == i + 1:
                hairpin_spans.append((i + 1, j - 1, []))

        def span_from_half_open(start: int, end: int) -> Tuple[int, int] | None:
            if start < end:
                return (start, end - 1)
            return None

        def zero_length_span_from_half_open(start: int, end: int) -> Tuple[int, int]:
            return (start, end - 1)

        open_to_close = loop_segments.open_to_close_map.to(dtype=torch.long).view(-1).cpu()
        if open_to_close.numel() > 0:
            roots, children, _ = compute_loop_spans(open_to_close, len(open_to_close))
            stack = roots[::-1]
            while stack:
                opener = stack.pop()
                closer = int(open_to_close[opener].item())
                child_openers = sorted(children[opener])
                if not child_openers:
                    pass
                elif len(child_openers) == 1:
                    child = child_openers[0]
                    child_close = int(open_to_close[child].item())
                    seg1 = span_from_half_open(opener + 1, child)
                    seg2 = span_from_half_open(child_close + 1, closer)
                    if seg1 and seg2:
                        spans: List[Tuple[int, int, List[int]]] = []
                        if seg1 in internal_spans:
                            knots_in = knots_for_span(seg1[0], seg1[1])
                            spans.append((seg1[0], seg1[1], knots_in))
                        if seg2 in internal_spans:
                            knots_in = knots_for_span(seg2[0], seg2[1])
                            spans.append((seg2[0], seg2[1], knots_in))
                        if len(spans) == 2:
                            spans.sort(key=lambda span: span[0])
                            internal_loops.append(spans)
                else:
                    spans: List[Tuple[int, int, List[int]]] = []  # type: ignore[no-redef]
                    prev = opener
                    for child in child_openers:
                        seg = zero_length_span_from_half_open(prev + 1, child)
                        knots_in = knots_for_span(seg[0], seg[1])
                        spans.append((seg[0], seg[1], knots_in))
                        prev = int(open_to_close[child].item())
                    seg = zero_length_span_from_half_open(prev + 1, closer)
                    knots_in = knots_for_span(seg[0], seg[1])
                    spans.append((seg[0], seg[1], knots_in))
                    if spans:
                        spans.sort(key=lambda span: span[0])
                        multi_loops.append(spans)
                stack.extend(reversed(child_openers))

        hairpin_spans.sort(key=lambda span: (span[0], span[1]))

        for h, (h_start, h_stop, knots_in) in enumerate(hairpin_spans, start=1):
            h_seq = self._slice_sequence(self.sequence, h_start, h_stop)
            pos5 = h_start - 1
            pos3 = h_stop + 1
            nuc1 = self._safe_base(self.sequence, pos5)
            nuc2 = self._safe_base(self.sequence, pos3)
            pseudoknot_label = f"PK{{{','.join(map(str, knots_in))}}}" if knots_in else ""
            label = f"H{h}"
            structure_types["H"].append(
                (
                    f'{label} {self._format_range(h_start, h_stop)} "{h_seq}" '
                    f"({self._format_pos(pos5)},{self._format_pos(pos3)}) "
                    f"{nuc1}:{nuc2} {pseudoknot_label}"
                ).rstrip()
                + "\n"  # noqa: W503
            )
            for knot_id in knots_in:
                pseudoknot_loop_segments.setdefault(knot_id, []).append((label, h_start, h_stop))

        bulge_loops.sort(key=lambda span: (span[0], span[1]))
        for b, (b_start, b_stop, knots_in) in enumerate(bulge_loops, start=1):
            b_seq = self._slice_sequence(self.sequence, b_start, b_stop)
            bp5_pos1 = b_start - 1
            bp3_pos1 = b_stop + 1
            bp5_pos2 = pair_map.get(bp5_pos1, -1)
            bp3_pos2 = pair_map.get(bp3_pos1, -1)
            bp5_nt1 = self._safe_base(self.sequence, bp5_pos1)
            bp3_nt1 = self._safe_base(self.sequence, bp3_pos1)
            bp5_nt2 = self._safe_base(self.sequence, bp5_pos2)
            bp3_nt2 = self._safe_base(self.sequence, bp3_pos2)
            pseudoknot_label = f"PK{{{','.join(map(str, knots_in))}}}" if knots_in else ""
            label = f"B{b}"
            structure_types["B"].append(
                (
                    f'{label} {self._format_range(b_start, b_stop)} "{b_seq}" '
                    f"({self._format_pos(bp5_pos1)},{self._format_pos(bp5_pos2)}) "
                    f"{bp5_nt1}:{bp5_nt2} "
                    f"({self._format_pos(bp3_pos1)},{self._format_pos(bp3_pos2)}) "
                    f"{bp3_nt1}:{bp3_nt2} {pseudoknot_label}"
                ).rstrip()
                + "\n"  # noqa: W503
            )
            for knot_id in knots_in:
                pseudoknot_loop_segments.setdefault(knot_id, []).append((label, b_start, b_stop))

        internal_loops.sort(key=lambda spans: spans[0][0] if spans else -1)
        for i, spans in enumerate(internal_loops, start=1):
            spans.sort(key=lambda span: span[0])
            for ip, (i_start, i_stop, knots_in) in enumerate(spans, start=1):
                i_seq = self._slice_sequence(self.sequence, i_start, i_stop)
                bp5_pos1 = i_start - 1
                bp5_pos2 = pair_map.get(bp5_pos1, -1)
                nuc5_1 = self._safe_base(self.sequence, bp5_pos1)
                nuc5_2 = self._safe_base(self.sequence, bp5_pos2)
                pseudoknot_label = f"PK{{{','.join(map(str, knots_in))}}}" if knots_in else ""
                label = f"I{i}.{ip}"
                structure_types["I"].append(
                    (
                        f'{label} {self._format_range(i_start, i_stop)} "{i_seq}" '
                        f"({self._format_pos(bp5_pos1)},"
                        f"{self._format_pos(bp5_pos2)}) "
                        f"{nuc5_1}:{nuc5_2} {pseudoknot_label}"
                    ).rstrip()
                    + "\n"  # noqa: W503
                )
                for knot_id in knots_in:
                    pseudoknot_loop_segments.setdefault(knot_id, []).append((label, i_start, i_stop))

        multi_loops.sort(key=lambda spans: spans[0][0] if spans else -1)
        for m, spans in enumerate(multi_loops, start=1):
            spans.sort(key=lambda span: span[0])
            for mp, (m_start, m_stop, knots_in) in enumerate(spans, start=1):
                m_seq = self._slice_sequence(self.sequence, m_start, m_stop)
                bp5_pos1 = m_start - 1
                bp3_pos1 = m_stop + 1
                bp5_pos2 = pair_map.get(bp5_pos1, -1)
                bp3_pos2 = pair_map.get(bp3_pos1, -1)
                nuc5_1 = self._safe_base(self.sequence, bp5_pos1)
                nuc5_2 = self._safe_base(self.sequence, bp5_pos2)
                nuc3_1 = self._safe_base(self.sequence, bp3_pos1)
                nuc3_2 = self._safe_base(self.sequence, bp3_pos2)
                pseudoknot_label = f"PK{{{','.join(map(str, knots_in))}}}" if knots_in else ""
                label = f"M{m}.{mp}"
                structure_types["M"].append(
                    (
                        f'{label} {self._format_range(m_start, m_stop)} "{m_seq}" '
                        f"({self._format_pos(bp5_pos1)},"
                        f"{self._format_pos(bp5_pos2)}) "
                        f"{nuc5_1}:{nuc5_2} "
                        f"({self._format_pos(bp3_pos1)},"
                        f"{self._format_pos(bp3_pos2)}) "
                        f"{nuc3_1}:{nuc3_2} {pseudoknot_label}"
                    ).rstrip()
                    + "\n"  # noqa: W503
                )
                for knot_id in knots_in:
                    pseudoknot_loop_segments.setdefault(knot_id, []).append((label, m_start, m_stop))

        e_count = 0
        end_loops.sort(key=lambda span: span[0])
        for e_start, e_stop, knots_in in end_loops:
            if e_start > e_stop:
                continue
            e_count += 1
            label = f"E{e_count}"
            e_seq = self._slice_sequence(self.sequence, e_start, e_stop)
            pseudoknot_label = f"PK{{{','.join(map(str, knots_in))}}}" if knots_in else ""
            structure_types["E"].append(
                (f'{label} {self._format_range(e_start, e_stop)} "{e_seq}" ' f"{pseudoknot_label}").rstrip() + "\n"
            )
            for knot_id in knots_in:
                pseudoknot_loop_segments.setdefault(knot_id, []).append((label, e_start, e_stop))

        x_count = 0
        external_spans.sort(key=lambda span: span[0])
        for x_start, x_stop, knots_in in external_spans:
            if x_start > x_stop:
                continue
            x_count += 1
            x_seq = self._slice_sequence(self.sequence, x_start, x_stop)
            bp5_pos1 = x_start - 1
            bp3_pos1 = x_stop + 1
            bp5_pos2 = pair_map.get(bp5_pos1, -1)
            bp3_pos2 = pair_map.get(bp3_pos1, -1)
            nuc5_1 = self._safe_base(self.sequence, bp5_pos1)
            nuc5_2 = self._safe_base(self.sequence, bp5_pos2)
            nuc3_1 = self._safe_base(self.sequence, bp3_pos1)
            nuc3_2 = self._safe_base(self.sequence, bp3_pos2)
            pseudoknot_label = f"PK{{{','.join(map(str, knots_in))}}}" if knots_in else ""
            label = f"X{x_count}"
            structure_types["X"].append(
                (
                    f'{label} {self._format_range(x_start, x_stop)} "{x_seq}" '
                    f"({self._format_pos(bp5_pos1)},"
                    f"{self._format_pos(bp5_pos2)}) "
                    f"{nuc5_1}:{nuc5_2} "
                    f"({self._format_pos(bp3_pos1)},"
                    f"{self._format_pos(bp3_pos2)}) "
                    f"{nuc3_1}:{nuc3_2} {pseudoknot_label}"
                ).rstrip()
                + "\n"  # noqa: W503
            )
            for knot_id in knots_in:
                pseudoknot_loop_segments.setdefault(knot_id, []).append((label, x_start, x_stop))

        stem_label_by_key: Dict[Tuple[int, int, int, int], str] = {}
        stem_segments = self.topology.nested_helix_segments.segments
        for stem_count, helix_segment in enumerate(stem_segments, start=1):
            s_start1 = helix_segment.start_5p
            s_stop1 = helix_segment.stop_5p
            s_start2 = min(helix_segment.start_3p, helix_segment.stop_3p)
            s_stop2 = max(helix_segment.start_3p, helix_segment.stop_3p)
            label = f"S{stem_count}"
            s_seq1 = self._slice_sequence(self.sequence, s_start1, s_stop1)
            s_seq2 = self._slice_sequence(self.sequence, s_start2, s_stop2)
            structure_types["S"].append(
                f'{label} {self._format_range(s_start1, s_stop1)} "{s_seq1}" '
                f'{self._format_range(s_start2, s_stop2)} "{s_seq2}"\n'
            )
            stem_label_by_key[helix_segment.key] = label

        helix_pair_map = self.topology.nested_helix_segments.pair_map
        bp_pairs = [
            (i, j)
            for i, j in pair_map.pairs
            if (i, j) not in pseudoknot_pair_set and i not in pseudoknot_positions and j not in pseudoknot_positions
        ]
        sequence_for_ncbp = self.sequence
        if any(base.islower() for base in sequence_for_ncbp):
            sequence_for_ncbp = "".join(base.upper() for base in sequence_for_ncbp)
        noncanonical_primary = noncanonical_pairs_set(bp_pairs, sequence_for_ncbp, unsafe=True)
        ncbp_count = 0
        for i, j in bp_pairs:
            if (i, j) in noncanonical_primary:
                b1 = self._safe_base(self.sequence, i)
                b2 = self._safe_base(self.sequence, j)
                helix_segment = helix_pair_map.get((i, j))  # type: ignore[assignment]
                if helix_segment is None:
                    raise ValueError(f"No stem segment found for non-canonical pair {i},{j}")
                label = stem_label_by_key.get(helix_segment.key, "")
                if not label:
                    raise ValueError(f"No label found for non-canonical pair {i},{j}")
                ncbp_count += 1
                structure_types["NCBP"].append(
                    f"NCBP{ncbp_count} {self._format_pos(i)} {b1} " f"{self._format_pos(j)} {b2} {label}\n"
                )

        if pseudoknot_segments:
            for knot_idx, knot in enumerate(pseudoknot_segments):
                knot_id = knot_idx + 1
                knot_pairs: List[Pair] = list(knot.pairs)  # type: ignore[no-redef]
                knot_size = len(knot_pairs)
                k_5p_start = knot.start_5p
                k_3p_start = knot.start_3p
                k_5p_stop = knot.stop_5p
                k_3p_stop = knot.stop_3p
                linked_loops = ""
                loops = pseudoknot_loop_segments.get(knot_id, [])
                if len(loops) == 2:
                    loops_sorted = sorted(loops, key=lambda item: item[1])
                    (l_type1, l_start1, l_stop1), (l_type2, l_start2, l_stop2) = loops_sorted
                    linked_loops = (
                        f"{l_type1} {self._format_range(l_start1, l_stop1)} "
                        f"{l_type2} {self._format_range(l_start2, l_stop2)}"
                    )
                structure_types["PK"].append(
                    (
                        f"PK{knot_id} {knot_size}bp {self._format_range(k_5p_start, k_5p_stop)} "
                        f"{self._format_range(k_3p_stop, k_3p_start)} {linked_loops}"
                    ).rstrip()
                    + "\n"  # noqa: W503
                )
                noncanonical_knot = noncanonical_pairs_set(knot_pairs, sequence_for_ncbp, unsafe=True)
                for n_idx, (k_5p, k_3p) in enumerate(knot_pairs, start=1):
                    b_5p = self._safe_base(self.sequence, k_5p)
                    b_3p = self._safe_base(self.sequence, k_3p)
                    structure_types["PKBP"].append(
                        f"PK{knot_id}.{n_idx} {self._format_pos(k_5p)} {b_5p} " f"{self._format_pos(k_3p)} {b_3p}\n"
                    )
                    if (k_5p, k_3p) in noncanonical_knot:
                        ncbp_count += 1
                        structure_types["NCBP"].append(
                            f"NCBP{ncbp_count} {self._format_pos(k_5p)} {b_5p} "
                            f"{self._format_pos(k_3p)} {b_3p} PK{knot_id}.{n_idx}\n"
                        )

        for seg_idx, segment_pairs in enumerate(segment_pairs_list, start=1):
            if not segment_pairs:
                continue
            seg_size = len(segment_pairs)
            seg5p_start, seg3p_start = segment_pairs[0]
            seg3p_stop, seg5p_stop = segment_pairs[-1]
            seg_seq1 = self._slice_sequence(self.sequence, seg5p_start, seg3p_stop)
            seg_seq2 = self._slice_sequence(self.sequence, seg5p_stop, seg3p_start)
            structure_types["SEGMENTS"].append(
                f"segment{seg_idx} {seg_size}bp {self._format_range(seg5p_start, seg3p_stop)} "
                f"{seg_seq1} {self._format_range(seg5p_stop, seg3p_start)} {seg_seq2}\n"
            )

        return structure_types


def annotate_structure(structure: RnaSecondaryStructureTopology) -> str:
    """
    Return a bpRNA-like structural annotation string (structure array) for this structure.

    Labels: S (stems), H (hairpin loops), B (bulge loops), I (internal loops), M (multiloops),
    X (external loops), E (end).
    """
    segment_data = BpRnaSecondaryStructureTopology(structure.sequence, topology=structure)
    return segment_data.structural_annotation


def annotate_function(structure: RnaSecondaryStructureTopology) -> str:
    """
    Return a bpRNA-like functional annotation string (knot/function array) for this structure.

    Labels: K for bases involved in pseudoknot pairs, N otherwise.
    """
    segment_data = BpRnaSecondaryStructureTopology(structure.sequence, topology=structure)
    return segment_data.functional_annotation


def structure_types(structure: RnaSecondaryStructureTopology) -> Dict[str, List[str]]:
    """
    Return bpRNA-style structure type lines for this structure.
    """
    segment_data = BpRnaSecondaryStructureTopology(structure.sequence, topology=structure)
    return segment_data.structure_types


def annotate(structure: RnaSecondaryStructureTopology) -> Tuple[str, str]:
    """
    Return both structural and functional annotations for this structure.
    """
    segment_data = BpRnaSecondaryStructureTopology(structure.sequence, topology=structure)
    return segment_data.structural_annotation, segment_data.functional_annotation


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from multimolecule.io import load, write_st

    path = Path(sys.argv[1])

    loaded = load(path)

    if isinstance(loaded, list):
        if len(loaded) != 1:
            raise ValueError("load() must return a single record when writing .st")
        record = write_st(loaded[0], path.with_suffix(".st"))
    else:
        record = write_st(loaded, path.with_suffix(".st"))
