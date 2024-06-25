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
from dataclasses import dataclass

import pandas as pd
import torch
from Bio import SeqIO

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1013)


def convert_dataset(convert_config):
    dataframe = pd.DataFrame.from_dict(
        [
            {
                "id": record.id,
                "name": record.name,
                "sequence": str(record.seq),
                "type": record.description.split()[1],
                "description": record.description,
            }
            for record in SeqIO.parse(convert_config.dataset_path, format="fasta")
        ]
    )
    save_dataset(convert_config, dataframe)


@dataclass
class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
