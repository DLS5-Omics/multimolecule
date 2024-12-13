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
from pathlib import Path

import danling as dl
import pandas as pd
import torch

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1016)


cols = ["name", "sequence", "reactivity", "seqpos", "class", "dataset"]


def convert_dataset_(df: pd.DataFrame):
    df.drop("seqpos", axis=1, inplace=True)
    df = df.rename(
        columns={
            "Class": "class",
            "Dataset": "dataset",
            "orig_seqpos": "seqpos",
        }
    )
    df = df.sort_values("name")
    df = df[cols]
    return df


def convert_dataset(convert_config):
    df = dl.load_pandas(convert_config.dataset_path)
    save_dataset(convert_config, convert_dataset_(df), filename="test.parquet")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)

    def post(self):
        if not self.output_path:
            dataset_name = Path(self.dataset_path).stem
            seq_length = dataset_name.split("_")[2][6:]
            self.output_path = os.path.basename(os.path.dirname(__file__)).replace("_", "-") + f".{seq_length}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
