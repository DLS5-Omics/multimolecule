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
from collections.abc import Mapping, Sequence
from pathlib import Path

import requests
import torch
from Bio import SeqIO
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)


def convert_rfam(fasta, clans: Mapping) -> Sequence:
    fasta = Path(fasta)
    accession = fasta.stem
    cm = fasta.parent.parent.joinpath("cm", f"{accession}.cm")
    if cm.exists():
        try:
            with cm.open(encoding="utf-8") as f:
                lines = f.read().splitlines()
                family = lines[1].split()[1]
        except Exception:
            family = requests.get(f"https://rfam.org/family/{accession}/id").text
    else:
        family = requests.get(f"https://rfam.org/family/{accession}/id").text
    sequences = [
        {
            "id": seq.id,
            "sequence": str(seq.seq.upper()),
            "family": family,
            "clan": clans.get(family),
            "description": seq.description,
        }
        for seq in SeqIO.parse(fasta, format="fasta")
    ]
    sequences.sort(key=lambda s: s["id"])
    return sequences


def convert_dataset(convert_config):
    fastas = get_files(os.path.join(convert_config.dataset_path, "fasta"))
    with open(os.path.join(convert_config.dataset_path, "Rfam.clanin"), encoding="utf-8") as f:
        families = {line.split()[0]: line.split()[1:] for line in f.read().splitlines()}
        clans = {clan: family for family, clans in families.items() for clan in clans}
    data = [i for fasta in tqdm(fastas, total=len(fastas)) for i in convert_rfam(fasta, clans)]
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    tag: str | None = None
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))

    def post(self):
        super().post()
        if self.tag is None:
            path, version = os.path.split(self.dataset_path)
            if os.path.split(path)[-1] == "rfam":
                self.tag = version


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
