# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path

import torch
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)
RNA_SS_data = namedtuple("RNA_SS_data", "seq ss_label length name pairs")


def convert_bpseq(bpseq):
    if isinstance(bpseq, str):
        bpseq = Path(bpseq)
    with open(bpseq) as f:
        lines = f.read().splitlines()
    lines = [[int(i) if i.isdigit() else i for i in j.split()] for j in lines]
    sequence, structure = [], ["."] * len(lines)
    for row in lines:
        index, nucleotide, paired_index = row
        sequence.append(nucleotide)
        if paired_index > 0 and index < paired_index:
            structure[index - 1] = "("
            structure[paired_index - 1] = ")"
    return {"id": bpseq.stem.split("-")[0], "sequence": "".join(sequence), "secondary_structure": "".join(structure)}


def convert_dataset(convert_config):
    data = [convert_bpseq(file) for file in tqdm(get_files(convert_config.dataset_path))]
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__)).replace("_", "-")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
