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
from collections.abc import Mapping
from pathlib import Path

import torch
from Bio import SeqIO
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)


def convert_rfam(cm: str, fasta: str) -> Mapping:
    assert Path(cm).stem == Path(fasta).stem
    with open(cm) as f:
        lines = f.read().splitlines()
    return {"id": lines[1].split()[1], "sequences": [str(s.seq) for s in SeqIO.parse(fasta, format="fasta")]}


def convert_dataset(convert_config):
    cms = get_files(os.path.join(convert_config.dataset_path, "cm"))
    fastas = get_files(os.path.join(convert_config.dataset_path, "fasta"))
    assert len(cms) == len(fastas)
    data = [convert_rfam(cm, fasta) for cm, fasta in tqdm(zip(cms, fastas), total=len(cms))]
    data.sort(key=lambda s: s["id"])
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
