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
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .utils.rna.secondary_structure import notations
from .utils.rna.secondary_structure.bprna import STRUCTURE_TYPE_KEYS


class InvalidStructureFile(ValueError):
    r"""Raised when an input structure file is malformed."""


DBN = ("db", "dbn")
BPSEQ = ("bpseq",)
FASTA = ("fa", "fasta", "fna")
ST = ("st", "sta")
SUPPORTED = DBN + BPSEQ + FASTA + ST


@dataclass
class RnaSecondaryStructureRecord:
    r"""Container for RNA secondary structure data (sequence + dot-bracket)."""

    sequence: str
    dot_bracket: str
    id: str | None = None

    def __post_init__(self) -> None:
        if len(self.sequence) != len(self.dot_bracket):
            raise ValueError("sequence and dot_bracket must have the same length")

    def __len__(self) -> int:
        return len(self.sequence)

    @classmethod
    def read_dbn(cls, path: str | Path) -> RnaSecondaryStructureRecord:
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
        return cls(sequence=sequence, dot_bracket=dot_bracket, id=id)

    @classmethod
    def read_bpseq(cls, path: str | Path) -> RnaSecondaryStructureRecord:
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
        return cls(sequence="".join(sequence), dot_bracket=dot_bracket, id=path.stem)

    @classmethod
    def read_st(cls, path: str | Path) -> RnaSecondaryStructureRecord:
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
        return cls(sequence=sequence, dot_bracket=dot_bracket, id=record_id)

    def write_dbn(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w") as fh:
            if self.id:
                fh.write(f">{self.id}\n")
            fh.write(f"{self.sequence}\n")
            fh.write(f"{self.dot_bracket}\n")
        return path

    def write_bpseq(self, path: str | Path) -> Path:
        path = Path(path)
        pairs = notations.dot_bracket_to_pairs(self.dot_bracket)
        pair_index = _pair_index_from_pairs(pairs, len(self))
        with path.open("w") as fh:
            for idx, base in enumerate(self.sequence):
                partner = pair_index[idx]
                out_partner = partner + 1 if partner != -1 else 0
                fh.write(f"{idx + 1} {base} {out_partner}\n")
        return path


@dataclass(kw_only=True)
class BpRnaRecord(RnaSecondaryStructureRecord):
    r"""Container for bpRNA .st annotations."""

    structure_array: str
    knot_array: str
    structure_types: Dict[str, List[str]]
    page_number: int | None

    @classmethod
    def from_rna_secondary_structure_record(cls, record: RnaSecondaryStructureRecord) -> BpRnaRecord:
        from .utils.rna.secondary_structure.bprna import BpRnaSecondaryStructureTopology

        pairs = notations.dot_bracket_to_pairs(record.dot_bracket)
        segment_data = BpRnaSecondaryStructureTopology(record.sequence, pairs, dot_bracket=record.dot_bracket)
        page_number = _count_bracket_tiers(record.dot_bracket)
        return cls(
            sequence=record.sequence,
            dot_bracket=record.dot_bracket,
            id=record.id,
            page_number=page_number,
            structure_array=segment_data.structural_annotation,
            knot_array=segment_data.functional_annotation,
            structure_types=segment_data.structure_types,
        )

    @classmethod
    def from_sequence_pairs(
        cls,
        sequence: str,
        pairs: np.ndarray | Sequence[tuple[int, int]],
        *,
        id: str | None = None,
        dot_bracket: str | None = None,
    ) -> BpRnaRecord:
        from .utils.rna.secondary_structure.bprna import BpRnaSecondaryStructureTopology
        from .utils.rna.secondary_structure.pairs import normalize_pairs

        pairs = np.asarray(normalize_pairs(pairs), dtype=int)
        if dot_bracket is None:
            dot_bracket = notations.pairs_to_dot_bracket(pairs, length=len(sequence), unsafe=True)
        elif len(dot_bracket) != len(sequence):
            raise ValueError("dot_bracket length must match sequence length")
        page_number = _count_bracket_tiers(dot_bracket)
        segment_data = BpRnaSecondaryStructureTopology(sequence, pairs, dot_bracket=dot_bracket)
        return cls(
            sequence=sequence,
            dot_bracket=dot_bracket,
            id=id,
            page_number=page_number,
            structure_array=segment_data.structural_annotation,
            knot_array=segment_data.functional_annotation,
            structure_types=segment_data.structure_types,
        )

    @classmethod
    def read_st(cls, path: str | Path) -> BpRnaRecord:
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
                lowered = line.lower()
                if lowered.startswith("#id:") or lowered.startswith("#name:"):
                    raise InvalidStructureFile(f"Multiple .st records found in {path!s}")
                continue
            key = _st_line_key(line)
            structure_types.setdefault(key, []).append(_ensure_newline(line))

        return cls(
            sequence=sequence,
            dot_bracket=dot_bracket,
            id=headers.get("id") or headers.get("name"),
            page_number=page_number,
            structure_array=structure_array,
            knot_array=knot_array,
            structure_types=structure_types,
        )

    def write_st(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w") as fh:
            id = self.id or path.stem
            pairs = notations.dot_bracket_to_pairs(self.dot_bracket)
            self.dot_bracket = notations.pairs_to_dot_bracket(pairs, length=len(self), unsafe=True)
            page_number = self.page_number if self.page_number is not None else 1

            fh.write(f"#Name: {id}\n")
            fh.write(f"#Length: {len(self)}\n")
            fh.write(f"#PageNumber: {page_number}\n")
            fh.write(f"{self.sequence}\n")
            fh.write(f"{self.dot_bracket}\n")
            fh.write(f"{self.structure_array}\n")
            fh.write(f"{self.knot_array}\n")

            for key in STRUCTURE_TYPE_KEYS:
                for item in self.structure_types.get(key, []):
                    fh.write(_ensure_newline(item))
            extra_keys = sorted(k for k in self.structure_types if k not in STRUCTURE_TYPE_KEYS)
            for key in extra_keys:
                for item in self.structure_types.get(key, []):
                    fh.write(_ensure_newline(item))
        return path


@dataclass
class SequenceRecord:
    r"""Container for FASTA records."""

    sequence: str
    id: str | None = None
    comment: str | None = None

    def __len__(self) -> int:
        return len(self.sequence)

    @classmethod
    def read_fasta(cls, path: str | Path) -> SequenceRecord:
        path = Path(path)
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
                    if saw_header and seq_parts:
                        return cls(sequence="".join(seq_parts), id=id, comment=comment)
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
            return cls(sequence="".join(seq_parts), id=id, comment=comment)
        if saw_header:
            raise InvalidStructureFile(f"Missing sequence line after defline in {path!s}")
        raise InvalidStructureFile(f"No FASTA records found in {path!s}")

    def write_fasta(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w") as fh:
            name = self.id or path.stem
            comment = self.comment or ""
            fh.write(f">{name}{' ' + comment if comment else ''}\n")
            fh.write(f"{self.sequence}\n")
        return path


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


def read_dbn(path: str | Path) -> RnaSecondaryStructureRecord:
    r"""Parse a dot-bracket (.db/.dbn) file and return a single record."""
    return RnaSecondaryStructureRecord.read_dbn(path)


def write_dbn(record, path: str | Path) -> Path:
    r"""Write a dot-bracket (.db/.dbn) file for a single record."""
    if not isinstance(record, RnaSecondaryStructureRecord):
        raise TypeError("record must be a RnaSecondaryStructureRecord")
    return record.write_dbn(path)


def read_fasta(path: str | Path) -> SequenceRecord:
    r"""Parse a FASTA file and return a single record."""
    return SequenceRecord.read_fasta(path)


def write_fasta(record, path: str | Path) -> Path:
    r"""Write a FASTA file for a single record."""
    return record.write_fasta(path)


def read_bpseq(path: str | Path) -> RnaSecondaryStructureRecord:
    r"""Parse a BPSEQ file and return a record with dot-bracket notation."""
    return RnaSecondaryStructureRecord.read_bpseq(path)


def write_bpseq(record, path: str | Path) -> Path:
    r"""Write a BPSEQ file from a record with dot-bracket notation."""
    if not isinstance(record, RnaSecondaryStructureRecord):
        raise TypeError("record must be a RnaSecondaryStructureRecord")
    return record.write_bpseq(path)


def read_st(path: str | Path) -> BpRnaRecord:
    r"""Parse a .st/.sta file and return a single record."""
    return BpRnaRecord.read_st(path)


def write_st(record: BpRnaRecord | RnaSecondaryStructureRecord, path: str | Path) -> Path:
    r"""Write a .st/.sta file for a single record."""
    if not isinstance(record, (BpRnaRecord, RnaSecondaryStructureRecord)):
        raise TypeError("record must be a RnaSecondaryStructureRecord or BpRnaRecord")
    if isinstance(record, RnaSecondaryStructureRecord) and not isinstance(record, BpRnaRecord):
        record = BpRnaRecord.from_rna_secondary_structure_record(record)
    return record.write_st(path)


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


def _normalize_format(format: str | None, path: Path) -> str:
    if format is None:
        if not path.suffix:
            raise ValueError(f"Unable to infer format from path {path!s}")
        format = path.suffix[1:]
    return format.lower().lstrip(".")


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
    from .utils.rna.secondary_structure.bprna import annotate
    from .utils.rna.secondary_structure.topology import RnaSecondaryStructureTopology

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
