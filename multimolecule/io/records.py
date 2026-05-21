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
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..utils.rna.secondary_structure import notations


class InvalidStructureFile(ValueError):
    r"""Raised when an input structure file is malformed."""


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
        from .rna_secondary_structure import read_dbn

        return read_dbn(path)

    @classmethod
    def read_bpseq(cls, path: str | Path) -> RnaSecondaryStructureRecord:
        from .rna_secondary_structure import read_bpseq

        return read_bpseq(path)

    @classmethod
    def read_st(cls, path: str | Path) -> RnaSecondaryStructureRecord:
        from .rna_secondary_structure import read_rna_secondary_structure_st

        return read_rna_secondary_structure_st(path)

    def write_dbn(self, path: str | Path) -> Path:
        from .rna_secondary_structure import write_dbn

        return write_dbn(self, path)

    def write_bpseq(self, path: str | Path) -> Path:
        from .rna_secondary_structure import write_bpseq

        return write_bpseq(self, path)


@dataclass(kw_only=True)
class BpRnaRecord(RnaSecondaryStructureRecord):
    r"""Container for bpRNA .st annotations."""

    structure_array: str
    knot_array: str
    structure_types: Dict[str, List[str]]
    page_number: int | None

    @classmethod
    def from_rna_secondary_structure_record(cls, record: RnaSecondaryStructureRecord) -> BpRnaRecord:
        from ..utils.rna.secondary_structure.bprna import BpRnaSecondaryStructureTopology
        from .rna_secondary_structure import _count_bracket_tiers

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
        from ..utils.rna.secondary_structure.bprna import BpRnaSecondaryStructureTopology
        from ..utils.rna.secondary_structure.pairs import normalize_pairs
        from .rna_secondary_structure import _count_bracket_tiers

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
        from .rna_secondary_structure import read_st

        return read_st(path)

    def write_st(self, path: str | Path) -> Path:
        from .rna_secondary_structure import write_st

        return write_st(self, path)


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
        from .fasta import read_fasta

        return read_fasta(path)

    @classmethod
    def read_fasta_records(cls, path: str | Path) -> list[SequenceRecord]:
        from .fasta import read_fasta_records

        return read_fasta_records(path)

    def write_fasta(self, path: str | Path) -> Path:
        from .fasta import write_fasta

        return write_fasta(self, path)
