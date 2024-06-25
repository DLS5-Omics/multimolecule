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

from multimolecule.datasets.bprna.bprna import convert_sta
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import copy_readme, get_files, push_to_hub, write_data

torch.manual_seed(1016)


def save_dataset(convert_config: ConvertConfig, data: Mapping, compression: str = "brotli", level: int = 4):
    root, output_path = convert_config.root, convert_config.output_path
    os.makedirs(output_path, exist_ok=True)
    for name, d in data.items():
        write_data(d, output_path, name + ".parquet", compression, level)
    copy_readme(root, output_path)
    push_to_hub(convert_config, output_path)


def _convert_dataset(dataset):
    files = get_files(dataset)
    return [convert_sta(file) for file in tqdm(files, total=len(files))]


def convert_dataset(convert_config):
    data = {
        "train": _convert_dataset(os.path.join(convert_config.dataset_path, "TR0")),
        "val": _convert_dataset(os.path.join(convert_config.dataset_path, "VL0")),
        "test": _convert_dataset(os.path.join(convert_config.dataset_path, "TS0")),
    }
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__)).replace("_", "-")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
