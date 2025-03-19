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
import torch

from multimolecule.data import dot_bracket_to_contact_map
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

# List of fields that contain lists which need to be padded
list_fields = [
    "reactivity",
    "errors_reactivity",
    "deg_pH10",
    "errors_deg_pH10",
    "deg_50C",
    "errors_deg_50C",
    "deg_Mg_pH10",
    "errors_deg_Mg_pH10",
    "deg_Mg_50C",
    "errors_deg_Mg_50C",
    "errors",
]


def drop_low_quality(df, min_signal_to_noise: float = 1.0):
    if "signal_to_noise" not in df.columns:
        return df
    rows = []
    for idx, row in df.iterrows():
        if row["signal_to_noise"] < min_signal_to_noise:
            rows.append(idx)
    return df.drop(rows)


def drop_structure_failure(df):
    if "secondary_structure" not in df.columns:
        return df
    rows = []
    for idx, row in df.iterrows():
        try:
            dot_bracket_to_contact_map(row["secondary_structure"])
        except ValueError:
            rows.append(idx)
    return df.drop(rows)


def pad_to_sequence_length(df, padding_value: float | None = None):
    df_padded = df.copy()
    for idx, row in df_padded.iterrows():
        seq_length = len(row.sequence)
        for field in list_fields:
            if field in row and isinstance(row[field], list):
                current_length = len(row[field])
                if current_length < seq_length:
                    padded_list = row[field] + [padding_value] * (seq_length - current_length)
                    df_padded.at[idx, field] = padded_list
                elif current_length > seq_length:
                    df_padded.at[idx, field] = row[field][:seq_length]
    return df_padded


def preprocess(df, min_signal_to_noise: float = 1.0):
    df = drop_low_quality(df, min_signal_to_noise)
    df = drop_structure_failure(df)
    df = pad_to_sequence_length(df)
    return df


def convert_dataset(convert_config):
    df = dl.load_pandas(convert_config.dataset_path)
    df.SN_filter = df.SN_filter.astype(bool)
    df = df.rename(columns={"ID": "id", "design_name": "design", "structure": "secondary_structure"})
    signal_cols = [c for c in df.columns if c.startswith("signal_to_noise_")]
    df["signal_to_noise"] = df[signal_cols].mean(axis=1)
    df = df.sort_values("id")
    ryos1 = df[df["RYOS"] == 1]
    ryos2 = df[df["RYOS"] == 2]
    data1 = {
        "train": preprocess(ryos1[ryos1["split"] == "public_train"][cols]),
        "validation": preprocess(ryos1[ryos1["split"] == "public_test"][cols]),
        "test": preprocess(ryos1[ryos1["split"] == "private_test"][cols]),
    }
    data2 = {
        "train": preprocess(ryos2[ryos2["split"] != "private_test"][cols]),
        "test": preprocess(ryos2[ryos2["split"] == "private_test"][cols]),
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
