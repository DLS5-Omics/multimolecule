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

import danling as dl
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
    "errors_reactivity",
    "signal_to_noise_reactivity",
    "deg_pH10",
    "errors_deg_pH10",
    "signal_to_noise_deg_pH10",
    "deg_50C",
    "errors_deg_50C",
    "signal_to_noise_deg_50C",
    "deg_Mg_pH10",
    "errors_deg_Mg_pH10",
    "signal_to_noise_deg_Mg_pH10",
    "deg_Mg_50C",
    "errors_deg_Mg_50C",
    "signal_to_noise_deg_Mg_50C",
    "SN_filter",
]


def convert_dataset(convert_config):
    df = dl.load_pandas(convert_config.dataset_path)
    df.SN_filter = df.SN_filter.astype(bool)
    df = df.rename(columns={"ID": "id", "design_name": "design", "structure": "secondary_structure"})
    df = df.sort_values("id")
    ryos1 = df[df["RYOS"] == 1]
    ryos2 = df[df["RYOS"] == 2]
    data1 = {
        "train": ryos1[ryos1["split"] == "public_train"][cols],
        "validation": ryos1[ryos1["split"] == "public_test"][cols],
        "test": ryos1[ryos1["split"] == "private_test"][cols],
    }
    data2 = {
        "train": ryos2[ryos2["split"] != "private_test"][cols],
        "test": ryos2[ryos2["split"] == "private_test"][cols],
    }
    repo_id, output_path = convert_config.repo_id, convert_config.output_path
    convert_config.repo_id, convert_config.output_path = repo_id + "-1", output_path + "-1"
    save_dataset(convert_config, data1)
    convert_config.repo_id, convert_config.output_path = repo_id + "-2", output_path + "-2"
    save_dataset(convert_config, data2)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
