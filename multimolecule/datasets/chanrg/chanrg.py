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

import pandas as pd
import torch
from chanfig import DefaultDict
from tqdm import tqdm

from multimolecule.datasets.bprna_1m.bprna_1m import convert_sta
from multimolecule.datasets.chanrg.utils import query
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import get_files, save_dataset

torch.manual_seed(1016)

clans = query("SELECT * FROM clan_membership").set_index("rfam_acc")["clan_acc"].to_dict()
rna_architecture = pd.read_csv(os.path.join(os.path.dirname(__file__), "RNArchitecture.csv")).set_index("RFAM#")
architectures = rna_architecture["Architecture"].to_dict()
super_families = rna_architecture["Clan"].to_dict()


def convert_family(family):
    files = get_files(os.path.join(family))
    sequences = []
    family = os.path.basename(family)
    clan = clans.get(family)
    architecture = architectures.get(family)
    super_family = super_families.get(family)
    for file in files:
        if not file.endswith(".st"):
            continue
        ret = convert_sta(file)
        ret["family"] = family
        ret["clan"] = clan
        ret["architecture"] = architecture
        ret["super_family"] = super_family
        sequences.append(ret)
    return split_family(sequences)


def split_family(sequences, seed=1016, val_ratio=0.1, test_ratio=0.1):
    if not sequences:
        return []

    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1.0")

    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")

    current_state = torch.get_rng_state()

    try:
        torch.manual_seed(seed)

        total_sequences = len(sequences)

        accessions = DefaultDict(list)
        for seq in sequences:
            accession = seq["id"].split("_")[0]
            accessions[accession].append(seq)

        single_accessions = []
        multi_accessions = []

        for _, seqs in accessions.items():
            if len(seqs) == 1:
                single_accessions.extend(seqs)
            else:
                multi_accessions.extend(seqs)

        train_sequences = multi_accessions.copy()

        val_size = int(val_ratio * total_sequences)
        test_size = int(test_ratio * total_sequences)

        available_single = len(single_accessions)
        if val_size + test_size > available_single:
            for seq in sequences:
                seq["split"] = "GenF"
            return sequences

        if single_accessions:
            indices = torch.randperm(len(single_accessions))
            shuffled_single = [single_accessions[i] for i in indices]

            val_sequences = shuffled_single[:val_size]
            test_sequences = shuffled_single[val_size : val_size + test_size]
            train_sequences.extend(shuffled_single[val_size + test_size :])
        else:
            val_sequences = []
            test_sequences = []

        for seq in train_sequences:
            seq["split"] = "Train"
        for seq in val_sequences:
            seq["split"] = "Validation"
        for seq in test_sequences:
            seq["split"] = "Test"

        return train_sequences + val_sequences + test_sequences

    finally:
        torch.set_rng_state(current_state)


def convert_dataset(convert_config):
    families = get_files(convert_config.dataset_path)
    data = [
        i
        for family in tqdm(families, total=len(families))
        for i in convert_family(os.path.join(convert_config.dataset_path, family))
    ]
    data = pd.DataFrame(data)
    unique_clans = sorted([i for i in data["clan"].unique() if i is not None])[:-1]
    new_clans = [clan for clan in unique_clans if data[data["clan"] == clan].super_family.iloc[0] is None]
    data.loc[data["clan"].isin(new_clans), "split"] = "GenC"
    data.loc[data["architecture"] == "complex unclassified", "split"] = "GenA"
    train = data[data["split"] == "Train"].sort_values(by=["split", "id"])
    validation = data[data["split"] == "Validation"].sort_values(by=["split", "id"])
    test = data[data["split"].isin(["Test", "GenA", "GenC", "GenF"])].sort_values(by=["split", "id"])
    save_dataset(convert_config, {"train": train, "validation": validation, "test": test})


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
