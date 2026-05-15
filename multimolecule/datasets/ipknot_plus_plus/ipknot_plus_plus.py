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

import os
from pathlib import Path

import torch
from Bio import AlignIO
from tqdm import tqdm

from multimolecule import io
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1016)


def _normalize_sequence(sequence: str) -> str:
    return "".join(base if base in "ACGU" else "N" for base in sequence.upper().replace("T", "U"))


def _normalize_alignment_sequence(sequence: str) -> str:
    return "".join("-" if base == "-" else _normalize_sequence(base) for base in sequence)


def _get_files(root: str | Path, suffix: str) -> list[Path]:
    return sorted(Path(root).glob(f"*{suffix}"), key=lambda path: path.name)


def convert_bpseq(file: str | Path) -> dict:
    file = Path(file)
    record = io.read_bpseq(file)
    sequence = _normalize_sequence(record.sequence)
    return {
        "id": file.stem,
        "sequence": sequence,
        "secondary_structure": record.dot_bracket,
        "aligned_ids": [file.stem],
        "aligned_sequences": [sequence],
    }


def convert_alignment(file: str | Path, bpseq_root: str | Path) -> dict:
    file = Path(file)
    record = convert_bpseq(Path(bpseq_root) / f"{file.stem}.bpseq")
    alignment = AlignIO.read(file, "clustal")
    record["aligned_ids"] = [sequence.id for sequence in alignment]
    record["aligned_sequences"] = [_normalize_alignment_sequence(str(sequence.seq)) for sequence in alignment]
    seed_sequence = record["aligned_sequences"][0].replace("-", "")
    if record["sequence"] != seed_sequence:
        raise ValueError(f"Seed sequence in {file!s} does not match {file.stem}.bpseq")
    return record


def convert_dataset(convert_config):
    root = Path(convert_config.dataset_path)
    bprna_1m = [
        convert_bpseq(file) for file in tqdm(_get_files(root / "from_bpRNA-1m", ".bpseq"), desc="from_bpRNA-1m")
    ]
    rfam_14_5 = [
        convert_bpseq(file) for file in tqdm(_get_files(root / "from_Rfam14.5", ".bpseq"), desc="from_Rfam14.5")
    ]
    rfam_14_5_ref = [
        convert_alignment(file, root / "from_Rfam14.5")
        for file in tqdm(_get_files(root / "from_Rfam14.5_ref", ".aln"), desc="from_Rfam14.5_ref")
    ]
    rfam_14_5_mafft = [
        convert_alignment(file, root / "from_Rfam14.5")
        for file in tqdm(_get_files(root / "from_Rfam14.5_mafft", ".aln"), desc="from_Rfam14.5_mafft")
    ]
    save_dataset(
        convert_config,
        {
            "bprna_1m": bprna_1m,
            "rfam_14_5": rfam_14_5,
            "rfam_14_5_ref": rfam_14_5_ref,
            "rfam_14_5_mafft": rfam_14_5_mafft,
        },
    )


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__)).replace("_", "-")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
