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

import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..utils.rna.secondary_structure import notations
from ..utils.rna.secondary_structure.bprna import STRUCTURE_TYPE_KEYS
from .records import BpRnaRecord, InvalidStructureFile, RnaSecondaryStructureRecord

DBN = ("db", "dbn")
BPSEQ = ("bpseq",)
ST = ("st", "sta")
CT = ("ct",)


def read_dbn(path: str | Path) -> RnaSecondaryStructureRecord:
    r"""Parse a dot-bracket (.db/.dbn) file and return a single record."""

    path = Path(path)
    lines = _read_lines(path, drop_comments=True)
    if not lines:
        raise InvalidStructureFile(f"No dot-bracket records found in {path!s}")
    idx = 0
    line = lines[idx]
    id = None
    if line.startswith(">"):
        id = line[1:].strip() or None
        idx += 1
        if idx >= len(lines):
            raise InvalidStructureFile(f"Missing sequence line after defline in {path!s}")
        line = lines[idx]
    sequence = line
    idx += 1
    if idx >= len(lines):
        raise InvalidStructureFile(f"Missing dot-bracket line after sequence in {path!s}")
    dot_line = lines[idx]
    dot_bracket = dot_line
    if " " in dot_bracket or "\t" in dot_bracket:
        terms = dot_bracket.split()
        if len(terms) >= 1 and len(terms[0]) == len(sequence):
            dot_bracket = terms[0]
        else:
            raise InvalidStructureFile(f"Dot-bracket line contains unexpected whitespace in {path!s}: {dot_line}")
    if len(sequence) != len(dot_bracket):
        raise InvalidStructureFile(f"Sequence and dot-bracket lengths differ in {path!s}")
    return RnaSecondaryStructureRecord(sequence=sequence, dot_bracket=dot_bracket, id=id)


def write_dbn(record: RnaSecondaryStructureRecord, path: str | Path) -> Path:
    r"""Write a dot-bracket (.db/.dbn) file for a single record."""

    if not isinstance(record, RnaSecondaryStructureRecord):
        raise TypeError("record must be a RnaSecondaryStructureRecord")
    path = Path(path)
    with path.open("w") as fh:
        if record.id:
            fh.write(f">{record.id}\n")
        fh.write(f"{record.sequence}\n")
        fh.write(f"{record.dot_bracket}\n")
    return path


def read_bpseq(path: str | Path) -> RnaSecondaryStructureRecord:
    r"""Parse a BPSEQ file and return a record with dot-bracket notation."""

    path = Path(path)
    rows = []
    for lineno, line in enumerate(_read_lines(path, drop_comments=True), 1):
        parts = line.split()
        if len(parts) != 3:
            raise InvalidStructureFile(f"Expected 3 columns in BPSEQ at line {lineno}: {line}")
        try:
            idx = int(parts[0]) - 1
            base = parts[1]
            pair = int(parts[2])
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidStructureFile(f"Invalid BPSEQ data at line {lineno}: {line}") from exc
        if idx < 0:
            raise InvalidStructureFile(f"Invalid index {idx + 1} at line {lineno}")
        pair_idx = pair - 1 if pair != 0 else -1
        rows.append((idx, base, pair_idx, lineno))

    if not rows:
        raise InvalidStructureFile(f"No BPSEQ records found in {path!s}")

    length = max(idx for idx, _, _, _ in rows) + 1
    sequence = [""] * length
    pair_by_index: Dict[int, int] = {}
    for idx, base, pair_idx, lineno in rows:
        if idx >= length:
            raise InvalidStructureFile(f"Index {idx + 1} out of bounds at line {lineno}")
        if sequence[idx]:
            raise InvalidStructureFile(f"Position {idx + 1} duplicated at line {lineno}")
        sequence[idx] = base
        if pair_idx == -1:
            continue
        if pair_idx < 0:
            raise InvalidStructureFile(f"Invalid pair index {pair_idx + 1} at line {lineno}")
        if idx == pair_idx:
            raise InvalidStructureFile(f"Position {idx + 1} paired to itself (line {lineno})")
        if pair_idx in pair_by_index and pair_by_index[pair_idx] != idx:
            raise InvalidStructureFile(
                f"Position {pair_idx + 1} paired to both {pair_by_index[pair_idx] + 1} and {idx + 1} "
                f"(line {lineno})"
            )
        if idx in pair_by_index:
            raise InvalidStructureFile(f"Position {idx + 1} paired multiple times (line {lineno})")
        pair_by_index[idx] = pair_idx

    if any(base == "" for base in sequence):
        raise InvalidStructureFile(f"Missing sequence positions in {path!s}")

    for i, j in pair_by_index.items():
        if pair_by_index.get(j) != i:
            raise InvalidStructureFile(f"Inconsistent pairing: {i + 1} -> {j + 1} but reverse not found")

    pairs = [(i, j) for i, j in pair_by_index.items() if i < j]
    pairs = np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)
    dot_bracket = notations.pairs_to_dot_bracket(pairs, length=length, unsafe=True)
    return RnaSecondaryStructureRecord(sequence="".join(sequence), dot_bracket=dot_bracket, id=path.stem)


def write_bpseq(record: RnaSecondaryStructureRecord, path: str | Path) -> Path:
    r"""Write a BPSEQ file from a record with dot-bracket notation."""

    if not isinstance(record, RnaSecondaryStructureRecord):
        raise TypeError("record must be a RnaSecondaryStructureRecord")
    path = Path(path)
    pairs = notations.dot_bracket_to_pairs(record.dot_bracket)
    pair_index = _pair_index_from_pairs(pairs, len(record))
    with path.open("w") as fh:
        for idx, base in enumerate(record.sequence):
            partner = pair_index[idx]
            out_partner = partner + 1 if partner != -1 else 0
            fh.write(f"{idx + 1} {base} {out_partner}\n")
    return path


def read_ct(path: str | Path) -> RnaSecondaryStructureRecord:
    r"""Parse a Connect Table (.ct) file and return a record with dot-bracket notation."""

    path = Path(path)
    lines = _read_lines(path, drop_comments=False)
    if not lines:
        raise InvalidStructureFile(f"No CT records found in {path!s}")

    header = lines[0].split(None, 1)
    try:
        count = int(header[0])
    except (IndexError, ValueError) as exc:
        raise InvalidStructureFile(f"Invalid CT header in {path!s}: {lines[0]}") from exc
    if count < 0:
        raise InvalidStructureFile(f"Invalid CT length {count} in {path!s}")
    title = header[1].strip() if len(header) > 1 else ""
    record_id = _ct_header_id(title) or path.stem

    data = lines[1:]
    if len(data) < count:
        raise InvalidStructureFile(f"CT header declares {count} positions but only {len(data)} rows found in {path!s}")

    sequence = [""] * count
    pair_by_index: Dict[int, int] = {}
    for lineno, line in enumerate(data[:count], 2):
        parts = line.split()
        if len(parts) < 5:
            raise InvalidStructureFile(f"Expected at least 5 columns in CT at line {lineno}: {line}")
        try:
            idx = int(parts[0]) - 1
            base = parts[1]
            pair = int(parts[4])
        except ValueError as exc:
            raise InvalidStructureFile(f"Invalid CT data at line {lineno}: {line}") from exc
        if idx < 0 or idx >= count:
            raise InvalidStructureFile(f"Index {idx + 1} out of bounds at line {lineno}")
        if sequence[idx]:
            raise InvalidStructureFile(f"Position {idx + 1} duplicated at line {lineno}")
        sequence[idx] = base
        if pair == 0:
            continue
        pair_idx = pair - 1
        if pair_idx < 0 or pair_idx >= count:
            raise InvalidStructureFile(f"Invalid pair index {pair} at line {lineno}")
        if idx == pair_idx:
            raise InvalidStructureFile(f"Position {idx + 1} paired to itself (line {lineno})")
        if idx in pair_by_index and pair_by_index[idx] != pair_idx:
            raise InvalidStructureFile(f"Position {idx + 1} paired multiple times (line {lineno})")
        pair_by_index[idx] = pair_idx

    if any(base == "" for base in sequence):
        raise InvalidStructureFile(f"Missing sequence positions in {path!s}")
    for i, j in pair_by_index.items():
        if pair_by_index.get(j) != i:
            raise InvalidStructureFile(f"Inconsistent pairing: {i + 1} -> {j + 1} but reverse not found")

    pairs = [(i, j) for i, j in pair_by_index.items() if i < j]
    pairs = np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)
    dot_bracket = notations.pairs_to_dot_bracket(pairs, length=count, unsafe=True)
    return RnaSecondaryStructureRecord(sequence="".join(sequence), dot_bracket=dot_bracket, id=record_id)


def write_ct(record: RnaSecondaryStructureRecord, path: str | Path) -> Path:
    r"""Write a Connect Table (.ct) file from a record with dot-bracket notation."""

    if not isinstance(record, RnaSecondaryStructureRecord):
        raise TypeError("record must be a RnaSecondaryStructureRecord")
    path = Path(path)
    pairs = notations.dot_bracket_to_pairs(record.dot_bracket)
    pair_index = _pair_index_from_pairs(pairs, len(record))
    id = record.id or path.stem
    length = len(record)
    with path.open("w") as fh:
        fh.write(f"{length} {id}\n")
        for idx, base in enumerate(record.sequence):
            partner = pair_index[idx]
            out_partner = partner + 1 if partner != -1 else 0
            next_idx = idx + 2 if idx + 1 < length else 0
            fh.write(f"{idx + 1} {base} {idx} {next_idx} {out_partner} {idx + 1}\n")
    return path


def _ct_header_id(title: str) -> str | None:
    r"""Extract a record id from a CT header title, stripping a leading energy descriptor.

    CT header titles vary by tool: ``ENERGY = -1.2  name`` (mfold/UNAFold), ``dG = -1.2 name``,
    or just ``name``. We drop a recognized energy term and keep the remainder as the id.
    """

    if not title:
        return None
    match = re.match(r"\s*(?:energy|dg|delta\s*g)\s*=\s*\S+\s*", title, flags=re.IGNORECASE)
    if match:
        title = title[match.end() :]
    title = title.strip()
    return title or None


def read_rna_secondary_structure_st(path: str | Path) -> RnaSecondaryStructureRecord:
    r"""Parse sequence and dot-bracket data from a bpRNA .st/.sta file."""

    path = Path(path)
    lines = _read_lines(path, drop_comments=False)
    if not lines:
        raise InvalidStructureFile(f"No .st records found in {path!s}")
    idx = 0
    record_id = None
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if not line.startswith("#"):
            break
        header = line[1:]
        if ":" in header:
            key, value = header.split(":", 1)
            key = key.strip().lower().replace(" ", "")
            if key in {"id", "name"}:
                record_id = value.strip() or None
        idx += 1
    if idx + 1 >= len(lines):
        raise InvalidStructureFile(f"Incomplete .st record in {path!s}")
    sequence = lines[idx].strip()
    dot_bracket = lines[idx + 1].strip()
    if len(sequence) != len(dot_bracket):
        raise InvalidStructureFile(f"Sequence and dot-bracket lengths differ in {path!s}")
    return RnaSecondaryStructureRecord(sequence=sequence, dot_bracket=dot_bracket, id=record_id)


def read_st(path: str | Path) -> BpRnaRecord:
    r"""Parse a .st/.sta file and return a single bpRNA record."""

    path = Path(path)
    lines = _read_lines(path, drop_comments=False)
    if not lines:
        raise InvalidStructureFile(f"No .st records found in {path!s}")

    headers: Dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if not line.startswith("#"):
            break
        header = line[1:]
        if ":" in header:
            key, value = header.split(":", 1)
            key = key.strip().lower().replace(" ", "")
            headers[key] = value.strip()
        idx += 1
    if idx + 3 >= len(lines):
        raise InvalidStructureFile(f"Incomplete .st record in {path!s}")

    sequence = lines[idx].strip()
    dot_bracket = lines[idx + 1].strip()
    structure_array = lines[idx + 2].strip()
    knot_array = lines[idx + 3].strip()
    idx += 4

    if len(sequence) != len(dot_bracket):
        raise InvalidStructureFile(f"Sequence and dot-bracket lengths differ in {path!s}")
    if len(sequence) != len(structure_array):
        raise InvalidStructureFile(f"Sequence and structure array lengths differ in {path!s}")
    if len(sequence) != len(knot_array):
        raise InvalidStructureFile(f"Sequence and knot array lengths differ in {path!s}")

    length_header = headers.get("length")
    if length_header:
        try:
            expected = int("".join(ch for ch in length_header if ch.isdigit()))
        except ValueError as exc:
            raise InvalidStructureFile(f"Invalid Length header in {path!s}") from exc
        if expected != len(sequence):
            raise InvalidStructureFile(f"Length header {expected} != sequence length {len(sequence)} in {path!s}")

    page_number = None
    page_header = headers.get("pagenumber")
    if page_header:
        try:
            page_number = int(page_header)
        except ValueError:
            page_number = None

    structure_types: Dict[str, List[str]] = {}
    for line in lines[idx:]:
        if not line.strip():
            continue
        if line.startswith("#"):
            lowered = line.lower()
            if lowered.startswith("#id:") or lowered.startswith("#name:"):
                raise InvalidStructureFile(f"Multiple .st records found in {path!s}")
            continue
        key = _st_line_key(line)
        structure_types.setdefault(key, []).append(_ensure_newline(line))

    return BpRnaRecord(
        sequence=sequence,
        dot_bracket=dot_bracket,
        id=headers.get("id") or headers.get("name"),
        page_number=page_number,
        structure_array=structure_array,
        knot_array=knot_array,
        structure_types=structure_types,
    )


def write_st(record: BpRnaRecord | RnaSecondaryStructureRecord, path: str | Path, **_: object) -> Path:
    r"""Write a .st/.sta file for a single bpRNA record."""

    if not isinstance(record, (BpRnaRecord, RnaSecondaryStructureRecord)):
        raise TypeError("record must be a RnaSecondaryStructureRecord or BpRnaRecord")
    if isinstance(record, RnaSecondaryStructureRecord) and not isinstance(record, BpRnaRecord):
        record = BpRnaRecord.from_rna_secondary_structure_record(record)
    path = Path(path)
    with path.open("w") as fh:
        id = record.id or path.stem
        pairs = notations.dot_bracket_to_pairs(record.dot_bracket)
        dot_bracket = notations.pairs_to_dot_bracket(pairs, length=len(record), unsafe=True)
        page_number = record.page_number if record.page_number is not None else 1

        fh.write(f"#Name: {id}\n")
        fh.write(f"#Length: {len(record)}\n")
        fh.write(f"#PageNumber: {page_number}\n")
        fh.write(f"{record.sequence}\n")
        fh.write(f"{dot_bracket}\n")
        fh.write(f"{record.structure_array}\n")
        fh.write(f"{record.knot_array}\n")

        for key in STRUCTURE_TYPE_KEYS:
            for item in record.structure_types.get(key, []):
                fh.write(_ensure_newline(item))
        extra_keys = sorted(k for k in record.structure_types if k not in STRUCTURE_TYPE_KEYS)
        for key in extra_keys:
            for item in record.structure_types.get(key, []):
                fh.write(_ensure_newline(item))
    return path


def _read_lines(path: str | Path, *, drop_comments: bool) -> List[str]:
    path = Path(path)
    lines: List[str] = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if drop_comments and stripped.startswith("#"):
                continue
            lines.append(stripped)
    return lines


def _pair_index_from_pairs(pairs: np.ndarray, length: int) -> List[int]:
    pair_index = [-1] * length
    for i, j in pairs.tolist():
        if i == j:
            continue
        if i < 0 or j < 0 or i >= length or j >= length:
            raise InvalidStructureFile("Pair indices out of bounds")
        if pair_index[i] not in (-1, j) or pair_index[j] not in (-1, i):
            raise InvalidStructureFile("Conflicting base pairs found")
        pair_index[i] = j
        pair_index[j] = i
    return pair_index


def _st_line_key(line: str) -> str:
    token = line.split()[0]
    if token.lower().startswith("segment"):
        return "SEGMENTS"
    if token.startswith("PKBP"):
        return "PKBP"
    if token.startswith("PK") and "." in token:
        return "PKBP"
    if token.startswith("PK"):
        return "PK"
    if token.startswith("NCBP"):
        return "NCBP"
    if token.startswith("SEGMENTS"):
        return "SEGMENTS"
    if token and token[0] in {"S", "H", "B", "I", "M", "X", "E"}:
        return token[0]
    return "UNKNOWN"


def _ensure_newline(line: str) -> str:
    return line if line.endswith("\n") else f"{line}\n"


def _auto_annotate(record: RnaSecondaryStructureRecord) -> tuple[str, str]:
    from ..utils.rna.secondary_structure.bprna import annotate
    from ..utils.rna.secondary_structure.topology import RnaSecondaryStructureTopology

    structure = RnaSecondaryStructureTopology(record.sequence, record.dot_bracket)
    return annotate(structure)


def _count_bracket_tiers(dot_bracket: str) -> int:
    openers = notations._DOT_BRACKET_PAIR_TABLE
    closers = notations._REVERSE_DOT_BRACKET_PAIR_TABLE
    used = set()
    for char in dot_bracket:
        if char in openers:
            used.add(char)
        elif char in closers:
            used.add(closers[char])
    return max(1, len(used))
