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

import torch
from tqdm import tqdm

from multimolecule.datasets.bprna.bprna import convert_sta
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)


def _convert_dataset(dataset):
    files = get_files(dataset)
    return [convert_sta(file) for file in tqdm(files, total=len(files))]


def convert_dataset(convert_config):
    data = {
        "train": _convert_dataset(os.path.join(convert_config.dataset_path, "TR0")),
        "validation": _convert_dataset(os.path.join(convert_config.dataset_path, "VL0")),
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
