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
from Bio import SeqIO
from tqdm import tqdm

from multimolecule.datasets.bprna_1m.bprna_1m import convert_sta
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset
from multimolecule.io import BpRnaRecord

torch.manual_seed(1016)


def _normalize_sequence(sequence: str) -> str:
    return "".join(base if base in "ACGU" else "N" for base in sequence.upper())


def _read_fasta(file: str | Path) -> tuple[str, str]:
    records = list(SeqIO.parse(file, "fasta"))
    if len(records) != 1:
        raise ValueError(f"Expected exactly one FASTA record in {file}, found {len(records)}.")
    record = records[0]
    return record.id, str(record.seq)


def _read_pairs(file: str | Path) -> list[tuple[int, int]]:
    pairs = []
    for line in Path(file).read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pairs.append((int(fields[0]) - 1, int(fields[1]) - 1))
    return pairs


def _drop_multipairs(pairs: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    used: set[int] = set()
    kept: list[tuple[int, int]] = []
    for i, j in pairs:
        if i == j or i < 0 or j < 0 or i >= length or j >= length or i in used or j in used:
            continue
        used.update((i, j))
        kept.append((i, j))
    return kept


def convert_sequence_label(sequence_file: str | Path, label_file: str | Path) -> dict:
    sequence_id, sequence = _read_fasta(sequence_file)
    label_id = Path(label_file).name
    if sequence_id != label_id:
        raise ValueError(f"Sequence ID {sequence_id!r} does not match label ID {label_id!r}.")
    sequence = _normalize_sequence(sequence)
    pairs = _drop_multipairs(_read_pairs(label_file), length=len(sequence))
    structure = BpRnaRecord.from_sequence_pairs(sequence, pairs, id=sequence_id)
    return {
        "id": sequence_id,
        "sequence": structure.sequence,
        "secondary_structure": structure.dot_bracket,
        "structural_annotation": structure.structure_array,
        "functional_annotation": structure.knot_array,
    }


def _convert_sequence_label_dataset(root: str, original_split: str) -> list[dict]:
    sequence_dir = Path(root) / f"{original_split}_sequences"
    label_dir = Path(root) / f"{original_split}_labels"
    label_files = [Path(file) for file in get_files(str(label_dir))]
    records: list[dict] = []
    for label_file in tqdm(label_files, total=len(label_files)):
        sequence_file = sequence_dir / label_file.name
        if not sequence_file.exists():
            raise FileNotFoundError(f"Missing sequence file for {label_file.name}: {sequence_file}")
        records.append(convert_sequence_label(sequence_file, label_file))
    return records


def _build_datasets(root: str) -> dict[str, dict[str, list[dict]]]:
    data0: dict[str, list[dict]] = {}
    for split, original_split in {"train": "TR0", "validation": "VL0", "test": "TS0"}.items():
        files = get_files(os.path.join(root, original_split))
        data0[split] = [convert_sta(file) for file in tqdm(files, total=len(files))]
    data1: dict[str, list[dict]] = {
        "train": _convert_sequence_label_dataset(root, "TR1"),
        "validation": _convert_sequence_label_dataset(root, "VL1"),
        "test": _convert_sequence_label_dataset(root, "TS1"),
    }
    data2: dict[str, list[dict]] = {
        "test": _convert_sequence_label_dataset(root, "TS2"),
    }
    composite: dict[str, list[dict]] = {
        "train": data0["train"] + data1["train"],
        "validation": data0["validation"] + data1["validation"],
        "test": data0["test"] + data1["test"] + data2["test"],
    }
    return {
        "": composite,
        "0": data0,
        "1": data1,
        "2": data2,
    }


def _save_variant(convert_config: ConvertConfig, suffix: str, data: dict[str, list[dict]]) -> None:
    output_path, repo_id = convert_config.output_path, convert_config.repo_id
    if suffix:
        convert_config.output_path = f"{output_path}-{suffix}"
        if repo_id is not None:
            convert_config.repo_id = f"{repo_id}-{suffix}"
    try:
        save_dataset(convert_config, data)
    finally:
        convert_config.output_path = output_path
        convert_config.repo_id = repo_id


def _strip_variant_suffix(value: str | None) -> str | None:
    if value is not None and value.endswith(("-0", "-1", "-2")):
        return value[:-2]
    return value


def convert_dataset(convert_config):
    datasets = _build_datasets(convert_config.dataset_path)
    output_path, repo_id = convert_config.output_path, convert_config.repo_id
    convert_config.output_path = _strip_variant_suffix(output_path)
    convert_config.repo_id = _strip_variant_suffix(repo_id)
    try:
        for suffix, data in datasets.items():
            if suffix:
                _save_variant(convert_config, suffix, data)
            else:
                save_dataset(convert_config, data)
    finally:
        convert_config.output_path = output_path
        convert_config.repo_id = repo_id


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__)).replace("_", "-")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
