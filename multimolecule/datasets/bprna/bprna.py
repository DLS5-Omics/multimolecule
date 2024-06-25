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

import torch
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)


def convert_sta(file: str) -> Mapping:
    with open(file) as f:
        lines = f.read().splitlines()
    idx = 0
    while lines[idx].startswith("#"):
        idx += 1
    return {
        "id": lines[0][7:],
        "sequence": lines[idx],
        "secondary_structure": lines[idx + 1],
        "structural_annotation": lines[idx + 2],
        "functional_annotation": lines[idx + 3],
    }


def convert_dataset(convert_config):
    files = get_files(convert_config.dataset_path)
    data = [convert_sta(file) for file in tqdm(files, total=len(files))]
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
