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

from .fasta import FASTA, read_fasta, write_fasta
from .rna_secondary_structure import (
    BPSEQ,
    CT,
    DBN,
    ST,
    read_bpseq,
    read_ct,
    read_dbn,
    read_st,
    write_bpseq,
    write_ct,
    write_dbn,
    write_st,
)

SUPPORTED = DBN + BPSEQ + FASTA + ST + CT


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
    if format in CT:
        return write_ct(record, path)
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
    if format in CT:
        return read_ct(path)
    raise ValueError(f"Trying to load {path!r} with unsupported extension={format!r}")


def _normalize_format(format: str | None, path: Path) -> str:
    if format is None:
        if not path.suffix:
            raise ValueError(f"Unable to infer format from path {path!s}")
        format = path.suffix[1:]
    return format.lower().lstrip(".")
