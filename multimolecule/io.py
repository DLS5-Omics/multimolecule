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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .utils.rna.secondary_structure import notations


class InvalidStructureFile(ValueError):
    r"""Raised when an input structure file is malformed."""


DBN = ("db", "dbn")
BPSEQ = ("bpseq",)
FASTA = ("fa", "fasta", "fna")
ST = ("st", "sta")
SUPPORTED = DBN + BPSEQ + FASTA + ST

_STRUCTURE_TYPE_ORDER = ("S", "H", "B", "I", "M", "X", "E", "PK", "PKBP", "NCBP", "SEGMENTS")


@dataclass
class StructureRecord:
    r"""Container for RNA secondary structure data and optional bpRNA annotations."""

    sequence: str
    dot_bracket: str
    id: str | None = None
    page_number: int | None = None
    structure_array: str | None = None
    knot_array: str | None = None
    structure_types: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.sequence) != len(self.dot_bracket):
            raise ValueError("sequence and dot_bracket must have the same length")

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass
class FastaRecord:
    r"""Container for FASTA records with optional dot-bracket annotations."""

    sequence: str
    id: str | None = None
    comment: str | None = None
    dot_bracket: str | None = None

    def __post_init__(self) -> None:
        if self.dot_bracket is not None and len(self.sequence) != len(self.dot_bracket):
            raise ValueError("sequence and dot_bracket must have the same length")

    def __len__(self) -> int:
        return len(self.sequence)


def save(record, path: str | Path, format: str | None = None, **kwargs) -> Path:
    r"""Save a structure file, dispatching by extension or explicit format."""
    path = Path(path)
    format = _normalize_format(format, path)
    if format in DBN:
        return write_dbn(record, path)
    if format in BPSEQ:
        return write_bpseq(record, path)
    if format in FASTA:
        return write_fasta(record, path)
    if format in ST:
        return write_st(record, path, **kwargs)
    raise ValueError(f"Trying to save {path!r} with unsupported extension={format!r}")


def load(path: str | Path, format: str | None = None):
    r"""Load a structure file, dispatching by extension or explicit format."""
    path = Path(path)
    format = _normalize_format(format, path)
    if format in DBN:
        return read_dbn(path)
    if format in BPSEQ:
        return read_bpseq(path)
    if format in FASTA:
        return read_fasta(path)
    if format in ST:
        return read_st(path)
    raise ValueError(f"Trying to load {path!r} with unsupported extension={format!r}")


def read_dbn(path: str | Path) -> StructureRecord | List[StructureRecord]:
    r"""Parse a dot-bracket (.db/.dbn) file and return one or many records."""
    records = _parse_dbn(path)
    if not records:
        raise InvalidStructureFile(f"No dot-bracket records found in {path!s}")
    if len(records) == 1:
        return records[0]
    return records


def write_dbn(record, path: str | Path) -> Path:
    r"""Write a dot-bracket (.db/.dbn) file for one or many records."""
    record_list = [_coerce_record(rec) for rec in _coerce_records(record)]
    path = Path(path)
    with path.open("w") as fh:
        for idx, item in enumerate(record_list):
            if idx:
                fh.write("\n")
            if item.id:
                fh.write(f">{item.id}\n")
            fh.write(f"{item.sequence}\n")
            fh.write(f"{item.dot_bracket}\n")
    return path


def read_fasta(path: str | Path) -> FastaRecord | List[FastaRecord]:
    r"""Parse a FASTA file and return one or many records.

    Dot-bracket annotations are inferred from header tokens when present.
    """
    parsed = _parse_fasta(path)
    if not parsed:
        raise InvalidStructureFile(f"No FASTA records found in {path!s}")
    if len(parsed) == 1:
        id, comment, sequence = parsed[0]
        return _fasta_record(id, comment, sequence)
    return [_fasta_record(id, comment, sequence) for id, comment, sequence in parsed]


def write_fasta(record, path: str | Path) -> Path:
    r"""Write a FASTA file for one or many records.

    If provided, dot-bracket annotations are appended to the header comment.
    """
    record_list = _coerce_fasta_records(record)
    path = Path(path)
    with path.open("w") as fh:
        for idx, item in enumerate(record_list):
            if idx:
                fh.write("\n")
            name = item.id or f"{path.stem}_{idx + 1}"
            header = _format_fasta_header(item, default_name=name)
            if header:
                fh.write(f"{header}\n")
            fh.write(f"{item.sequence}\n")
    return path


def write_fastas(records, path: str | Path) -> Path:
    r"""Alias for write_fasta (accepts one or many records)."""
    return write_fasta(records, path)


def read_bpseq(path: str | Path) -> StructureRecord:
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
                f"Position {pair_idx + 1} paired to both {pair_by_index[pair_idx] + 1} and {idx + 1} (line {lineno})"
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
    pairs_np = np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)
    dot_bracket = notations.pairs_to_dot_bracket(pairs_np, length=length)
    return StructureRecord(sequence="".join(sequence), dot_bracket=dot_bracket, id=path.stem)


def write_bpseq(record, path: str | Path) -> Path:
    r"""Write a BPSEQ file from a record with dot-bracket notation."""
    record = _coerce_record(record)
    path = Path(path)
    pairs = notations.dot_bracket_to_pairs(record.dot_bracket)
    pair_index = _pair_index_from_pairs(pairs, len(record))
    with path.open("w") as fh:
        for idx, base in enumerate(record.sequence):
            partner = pair_index[idx]
            out_partner = partner + 1 if partner != -1 else 0
            fh.write(f"{idx + 1} {base} {out_partner}\n")
    return path


def read_st(path: str | Path) -> StructureRecord | List[StructureRecord]:
    r"""Parse a .st/.sta file and return one or many records."""
    path = Path(path)
    lines = _read_lines(path, drop_comments=False)
    entries = _split_st_entries(lines)
    if not entries:
        raise InvalidStructureFile(f"No .st records found in {path!s}")
    if len(entries) == 1:
        return _parse_st_entry(entries[0], path)
    return [_parse_st_entry(entry, path) for entry in entries]


def write_st(record, path: str | Path, *, auto_annotation: bool = True) -> Path:
    r"""Write a .st/.sta file for one or many records.

    When auto_annotation is True, missing arrays are computed automatically.
    """
    path = Path(path)
    record_list = [_coerce_record(rec) for rec in _coerce_records(record)]
    _write_st_records(path, record_list, auto_annotation=auto_annotation)
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


def _parse_dbn(path: str | Path) -> List[StructureRecord]:
    lines = _read_lines(path, drop_comments=True)
    records: List[StructureRecord] = []
    idx = 0
    path = Path(path)
    while idx < len(lines):
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
        idx += 1
        dot_bracket = _parse_dot_bracket_line(dot_line, sequence, path)
        records.append(StructureRecord(sequence=sequence, dot_bracket=dot_bracket, id=id))
    return records


def _parse_fasta(path: str | Path) -> List[tuple[str | None, str | None, str]]:
    path = Path(path)
    records: List[tuple[str | None, str | None, str]] = []
    id: str | None = None
    comment: str | None = None
    seq_parts: List[str] = []
    saw_header = False
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith(";"):
                continue
            if stripped.startswith(">"):
                if seq_parts:
                    records.append((id, comment, "".join(seq_parts)))
                    seq_parts = []
                saw_header = True
                header = stripped[1:].strip()
                if header:
                    parts = header.split(None, 1)
                    id = parts[0]
                    comment = parts[1].strip() if len(parts) > 1 else None
                else:
                    id = None
                    comment = None
                continue
            seq_parts.append(stripped)
    if seq_parts:
        records.append((id, comment, "".join(seq_parts)))
    elif saw_header:
        raise InvalidStructureFile(f"Missing sequence line after defline in {path!s}")
    return records


def _normalize_format(format: str | None, path: Path) -> str:
    if format is None:
        if not path.suffix:
            raise ValueError(f"Unable to infer format from path {path!s}")
        format = path.suffix[1:]
    return format.lower().lstrip(".")


def _parse_dot_bracket_line(line: str, sequence: str, path: Path) -> str:
    dot_bracket = line
    if " " in dot_bracket or "\t" in dot_bracket:
        terms = dot_bracket.split()
        if len(terms) >= 1 and len(terms[0]) == len(sequence):
            dot_bracket = terms[0]
        else:
            raise InvalidStructureFile(f"Dot-bracket line contains unexpected whitespace in {path!s}: {line}")
    if len(sequence) != len(dot_bracket):
        raise InvalidStructureFile(f"Sequence and dot-bracket lengths differ in {path!s}")
    return dot_bracket


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


def _fasta_record(id: str | None, comment: str | None, sequence: str) -> FastaRecord:
    dot_bracket = _extract_dot_bracket(comment, len(sequence))
    return FastaRecord(sequence=sequence, id=id, comment=comment, dot_bracket=dot_bracket)


def _extract_dot_bracket(comment: str | None, length: int) -> str | None:
    if not comment:
        return None
    for token in comment.split():
        candidates = [token]
        if "=" in token:
            candidates.append(token.split("=", 1)[1])
        if ":" in token:
            candidates.append(token.split(":", 1)[1])
        for candidate in candidates:
            if len(candidate) != length:
                continue
            with suppress(ValueError):
                notations.dot_bracket_to_pairs(candidate)
                return candidate
    return None


def _format_fasta_header(record: FastaRecord, default_name: str) -> str:
    id = record.id or default_name
    comment = record.comment or ""
    dot_bracket = record.dot_bracket
    if dot_bracket:
        detected = _extract_dot_bracket(comment, len(dot_bracket))
        if detected != dot_bracket:
            comment = f"{comment} {dot_bracket}".strip() if comment else dot_bracket
    return f">{id}{' ' + comment if comment else ''}"


def _split_st_entries(lines: Sequence[str]) -> List[List[str]]:
    entries: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        lowered = line.lower()
        is_header = lowered.startswith("#id:") or lowered.startswith("#name:")
        if current and is_header:
            entries.append(current)
            current = []
        if current or is_header:
            current.append(line)
    if current:
        entries.append(current)
    return entries


def _parse_st_entry(lines: Sequence[str], path: Path) -> StructureRecord:
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
            expected = int(length_header)
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
            continue
        key = _st_line_key(line)
        structure_types.setdefault(key, []).append(_ensure_newline(line))

    return StructureRecord(
        sequence=sequence,
        dot_bracket=dot_bracket,
        id=headers.get("id") or headers.get("name"),
        page_number=page_number,
        structure_array=structure_array,
        knot_array=knot_array,
        structure_types=structure_types,
    )


def _st_line_key(line: str) -> str:
    token = line.split()[0]
    if token.startswith("PKBP"):
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


def _write_st_records(path: Path, records: Sequence[StructureRecord], *, auto_annotation: bool = True) -> None:
    with path.open("w") as fh:
        for idx, record in enumerate(records):
            if idx:
                fh.write("\n")
            id = record.id or path.stem
            page_number = record.page_number if record.page_number is not None else 1
            structure_array = record.structure_array
            knot_array = record.knot_array
            if (structure_array is None or knot_array is None) and auto_annotation:
                structure_array, knot_array = _auto_annotate(record)
            if structure_array is None or knot_array is None:
                raise InvalidStructureFile("structure_array and knot_array are required to write .st files")

            fh.write(f"#Name: {id}\n")
            fh.write(f"#Length: {len(record)}\n")
            fh.write(f"#PageNumber: {page_number}\n")
            fh.write(f"{record.sequence}\n")
            fh.write(f"{record.dot_bracket}\n")
            fh.write(f"{structure_array}\n")
            fh.write(f"{knot_array}\n")

            structure_types = record.structure_types or {}
            for key in _STRUCTURE_TYPE_ORDER:
                for item in structure_types.get(key, []):
                    fh.write(_ensure_newline(item))
            extra_keys = sorted(k for k in structure_types if k not in _STRUCTURE_TYPE_ORDER)
            for key in extra_keys:
                for item in structure_types.get(key, []):
                    fh.write(_ensure_newline(item))


def _auto_annotate(record: StructureRecord) -> tuple[str, str]:
    from .utils.rna.secondary_structure.bprna import annotate
    from .utils.rna.secondary_structure.topology import RnaSecondaryStructure

    structure = RnaSecondaryStructure(record.sequence, record.dot_bracket)
    return annotate(structure)


def _coerce_record(record) -> StructureRecord:
    if isinstance(record, StructureRecord):
        return record
    if isinstance(record, Mapping):
        return StructureRecord(**record)
    if isinstance(record, tuple) and len(record) == 2:
        return StructureRecord(sequence=record[0], dot_bracket=record[1])
    raise TypeError("record must be a StructureRecord, mapping, or (sequence, dot_bracket) tuple")


def _coerce_records(records) -> Iterable:
    if isinstance(records, StructureRecord):
        return [records]
    if isinstance(records, Mapping):
        return [records]
    if isinstance(records, tuple) and len(records) == 2 and isinstance(records[0], str):
        return [records]
    return records


def _coerce_fasta_record(record) -> FastaRecord:
    if isinstance(record, FastaRecord):
        return record
    if isinstance(record, StructureRecord):
        return FastaRecord(sequence=record.sequence, id=record.id, dot_bracket=record.dot_bracket)
    if isinstance(record, Mapping):
        if "sequence" not in record:
            raise TypeError("record mapping must include 'sequence'")
        sequence = record["sequence"]
        id = record.get("id")
        comment = record.get("comment")
        dot_bracket = record.get("dot_bracket")
        return FastaRecord(sequence=sequence, id=id, comment=comment, dot_bracket=dot_bracket)
    if isinstance(record, str):
        return FastaRecord(sequence=record)
    if isinstance(record, tuple) and len(record) == 2:
        id, sequence = record
        if not isinstance(sequence, str):
            raise TypeError("record tuple must be (id, sequence)")
        return FastaRecord(sequence=sequence, id=id if isinstance(id, str) else None)
    raise TypeError("record must be a FastaRecord, mapping, or (id, sequence) tuple")


def _coerce_fasta_records(records) -> List[FastaRecord]:
    if isinstance(records, (FastaRecord, StructureRecord, Mapping, str)):
        return [_coerce_fasta_record(records)]
    if isinstance(records, tuple) and len(records) == 2:
        return [_coerce_fasta_record(records)]
    return [_coerce_fasta_record(record) for record in records]
