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
from tqdm import tqdm

from multimolecule.datasets.bprna_new.bprna_new import convert_bpseq
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)


def _convert_dataset(root):
    files = get_files(root)
    return [convert_bpseq(file) for file in tqdm(files, total=len(files))]


def convert_dataset(convert_config):
    root = convert_config.dataset_path
    train_a = _convert_dataset(os.path.join(root, "TrainSetA"))
    train_b = _convert_dataset(os.path.join(root, "TrainSetB"))
    test_a = _convert_dataset(os.path.join(root, "TestSetA"))
    test_b = _convert_dataset(os.path.join(root, "TestSetB"))
    output_path, repo_id = convert_config.output_path, convert_config.repo_id
    save_dataset(convert_config, {"train": train_a, "validation": test_a, "test": test_b})
    convert_config.output_path = output_path + "-a"
    convert_config.repo_id = repo_id + "-a"
    save_dataset(convert_config, {"train": train_a, "test": test_a})
    convert_config.output_path = output_path + "-b"
    convert_config.repo_id = repo_id + "-b"
    save_dataset(convert_config, {"train": train_b, "test": test_b})


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
