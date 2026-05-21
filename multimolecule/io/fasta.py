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

from pathlib import Path
from typing import Sequence

from .records import InvalidStructureFile, SequenceRecord

FASTA = ("fa", "fas", "fasta", "fna", "ffn", "frn", "faa")


def read_fasta(path: str | Path) -> SequenceRecord:
    r"""Parse a FASTA file and return the first record."""

    return read_fasta_records(path)[0]


def read_fasta_records(path: str | Path) -> list[SequenceRecord]:
    r"""Parse a FASTA file and return all records."""

    path = Path(path)
    records: list[SequenceRecord] = []
    id: str | None = None
    comment: str | None = None
    seq_parts: list[str] = []
    saw_header = False

    def flush_record() -> None:
        nonlocal id, comment, seq_parts, saw_header
        if not saw_header and not seq_parts:
            return
        if saw_header and not seq_parts:
            raise InvalidStructureFile(f"Missing sequence line after defline in {path!s}")
        records.append(SequenceRecord(sequence="".join(seq_parts), id=id, comment=comment))
        id = None
        comment = None
        seq_parts = []
        saw_header = False

    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith(";"):
                continue
            if stripped.startswith(">"):
                flush_record()
                saw_header = True
                header = stripped[1:].strip()
                if header:
                    parts = header.split(None, 1)
                    id = parts[0]
                    comment = parts[1].strip() if len(parts) > 1 else None
                continue
            seq_parts.append(stripped)

    flush_record()
    if not records:
        raise InvalidStructureFile(f"No FASTA records found in {path!s}")
    return records


def write_fasta(record: SequenceRecord, path: str | Path) -> Path:
    r"""Write a FASTA file for a single record."""

    if not isinstance(record, SequenceRecord):
        raise TypeError("record must be a SequenceRecord")
    return write_fasta_records([record], path)


def write_fasta_records(records: Sequence[SequenceRecord], path: str | Path) -> Path:
    r"""Write a FASTA file for one or more records."""

    path = Path(path)
    if not records:
        raise ValueError("records must contain at least one SequenceRecord")
    with path.open("w") as fh:
        for index, record in enumerate(records):
            if not isinstance(record, SequenceRecord):
                raise TypeError("records must contain only SequenceRecord instances")
            name = record.id
            if name is None:
                name = path.stem if len(records) == 1 else f"{path.stem}_{index + 1}"
            comment = record.comment or ""
            fh.write(f">{name}{' ' + comment if comment else ''}\n")
            fh.write(f"{record.sequence}\n")
    return path
