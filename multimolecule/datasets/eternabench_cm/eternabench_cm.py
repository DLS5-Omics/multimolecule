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

import danling as dl
import pandas as pd
import torch

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1016)

cols = [
    "id",
    "design",
    "sequence",
    "secondary_structure",
    "reactivity",
    "errors",
    "signal_to_noise",
]


def convert_dataset_(df: pd.DataFrame):
    df.signal_to_noise = df.signal_to_noise.str.split(":").str[-1].astype(float)
    df = df.rename(columns={"ID": "id", "design_name": "design", "structure": "secondary_structure"})
    df = df.sort_values("id")
    df = df[cols]
    return df


def convert_dataset(convert_config):
    train = dl.load_pandas(convert_config.train_path)
    test = dl.load_pandas(convert_config.test_path)
    save_dataset(convert_config, {"train": convert_dataset_(train), "test": convert_dataset_(test)})


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__)).replace("_", "-")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
