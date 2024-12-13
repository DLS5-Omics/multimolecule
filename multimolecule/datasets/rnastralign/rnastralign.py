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
from collections.abc import Mapping
from pathlib import Path

import torch
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1016)


def convert_ct(file, family: str) -> Mapping:
    if not isinstance(file, Path):
        file = Path(file)
    with open(file) as f:
        lines = f.read().splitlines()

    first_line = lines[0].strip().split()
    num_bases = int(first_line[0])

    sequence = []
    dot_bracket = ["."] * num_bases

    # `N` does not exist in the ct files, so we need to add it
    if len(lines) < num_bases + 1:
        for i in range(1, num_bases + 1):
            if i >= len(lines):
                lines.append(f"{i} N {i-1} {i+1} 0 i")  # noqa: E226
            if int(lines[i].strip().split()[0]) != i:
                lines.insert(i, f"{i} N {i-1} {i+1} 0 i")  # noqa: E226

    for i in range(1, num_bases + 1):
        line = lines[i].strip().split()
        if int(line[0]) != i:
            raise ValueError(f"Invalid nucleotide index at position {i}: {line[0]} does not match the expected index.")
        sequence.append(line[1])
        pair_index = int(line[4])

        if pair_index > 0:
            if int(lines[pair_index].strip().split()[4]) != i:
                raise ValueError(
                    f"Invalid pairing at position {i}: pair_index {pair_index} does not point back correctly."
                )
            if pair_index > i:
                dot_bracket[i - 1] = "("
                dot_bracket[pair_index - 1] = ")"

    parts = list(file.parts)
    parts = parts[parts.index(family + "_database") :]
    parts[0] = parts[0][:-9]
    parts[-1] = parts[-1][:-3]

    return {
        "id": "-".join(parts),
        "sequence": "".join(sequence),
        "secondary_structure": "".join(dot_bracket),
        "family": family,
        "subfamily": parts[1] if len(parts) == 3 else None,
    }


def _convert_dataset(family_dir, max_seq_len: int | None = None):
    family_dir = Path(family_dir)
    family = family_dir.stem[:-9]
    files = [os.path.join(family_dir, f) for f in os.listdir(family_dir) if f.endswith(".ct")]
    if not files:
        for subdir in family_dir.iterdir():
            if subdir.is_dir():
                files.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".ct")])
    files.sort(key=lambda f: ("".join(filter(str.isalpha, f)), int("".join(filter(str.isdigit, f)))))
    data = [convert_ct(file, family) for file in tqdm(files, total=len(files))]
    if max_seq_len is not None:
        data = [d for d in data if len(d["sequence"]) <= max_seq_len]
    return data


def convert_dataset(convert_config):
    max_seq_len = convert_config.max_seq_len
    families = [
        os.path.join(convert_config.dataset_path, f)
        for f in os.listdir(convert_config.dataset_path)
        if f.endswith("_database")
    ]
    families.sort()
    data = [i for family in families for i in _convert_dataset(family, max_seq_len)]
    save_dataset(convert_config, data, filename="train.parquet")


class ConvertConfig(ConvertConfig_):
    max_seq_len: int | None = None
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))

    def post(self):
        if self.max_seq_len is not None:
            self.output_path = f"{self.output_path}.{self.max_seq_len}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
