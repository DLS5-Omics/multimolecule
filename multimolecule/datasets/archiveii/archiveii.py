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


def convert_ct(file) -> Mapping:
    if not isinstance(file, Path):
        file = Path(file)
    with open(file) as f:
        lines = f.readlines()

    first_line = lines[0].strip().split()
    num_bases = int(first_line[0])

    sequence = []
    dot_bracket = ["."] * num_bases

    for i in range(1, num_bases + 1):
        line = lines[i].strip().split()
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

    family, name = file.stem.split("_", 1)
    if family in ("5s", "16s", "23s"):
        family = family.upper() + "_rRNA"
    elif family == "srp":
        family = family.upper()
    elif family == "grp1":
        family = "group_I_intron"
    elif family == "grp2":
        family = "group_II_intron"
    id = family + "-" + name

    return {
        "id": id,
        "sequence": "".join(sequence),
        "secondary_structure": "".join(dot_bracket),
        "family": family,
    }


def convert_dataset(convert_config):
    max_seq_len = convert_config.max_seq_len
    files = [
        os.path.join(convert_config.dataset_path, f)
        for f in os.listdir(convert_config.dataset_path)
        if f.endswith(".ct")
    ]
    files.sort()
    data = [convert_ct(file) for file in tqdm(files, total=len(files))]
    if max_seq_len is not None:
        data = [d for d in data if len(d["sequence"]) <= max_seq_len]
    save_dataset(convert_config, data, filename="test.parquet")


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
