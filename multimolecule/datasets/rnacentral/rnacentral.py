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

import torch
from Bio import SeqIO
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1016)


def convert_dataset(convert_config):
    max_seq_len = convert_config.max_seq_len
    data = [
        {
            "urs": record.id,
            "sequence": str(record.seq) if max_seq_len is None else str(record.seq)[:max_seq_len],
            "type": record.description.split()[1],
            "description": record.description,
        }
        for record in tqdm(SeqIO.parse(convert_config.dataset_path, format="fasta"))
    ]
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    max_seq_len: int | None = None
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))

    def post(self):
        if self.max_seq_len is not None:
            self.output_path = f"{self.output_path}-{self.max_seq_len}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
