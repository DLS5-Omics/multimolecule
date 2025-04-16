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
from collections.abc import Mapping, Sequence

import torch

from multimolecule.data.functional import contact_map_to_dot_bracket
from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset

torch.manual_seed(1016)


def convert_sequence(sequence: str, label: str, id: str) -> Mapping:
    with open(sequence) as f:
        seq_id, sequence = f.read().splitlines()
    if seq_id[1:] != id:
        raise ValueError(f"Sequence ID does not match for {id}")
    with open(label) as f:
        label = f.read().splitlines()  # type: ignore[assignment]

    label_info = label[0].split()
    if len(label_info) == 3:
        _, label_id, seq_len = label_info
        seq_len = int(seq_len.split("=")[-1])  # type: ignore[assignment]
    elif len(label_info) == 5:
        _, label_id, _, _, seq_len = label_info
        seq_len = int(seq_len)  # type: ignore[assignment]
    else:
        raise ValueError(f"Label format is incorrect for {id}")
    if label_id != id:
        raise ValueError(f"Label ID does not match for {id}")
    if seq_len != len(sequence):
        raise ValueError(f"Sequence length does not match for {id}")
    if label[1].split() != ["i", "j"]:
        raise ValueError(f"Label format is incorrect for {id}")
    label = label[2:]
    contact_map = torch.zeros(seq_len, seq_len)
    for l in label:  # noqa: E741
        i, j = l.split()
        contact_map[int(i) - 1, int(j) - 1] = 1
    dot_bracket = contact_map_to_dot_bracket(contact_map, unsafe=True)
    return {
        "id": id,
        "sequence": sequence,
        "secondary_structure": dot_bracket,
    }


def convert_pdb_rna_secondary_structure(root: str, name: str) -> Sequence[Mapping]:
    sequence = os.path.join(root, name + "_sequences")
    label = os.path.join(root, name + "_labels")
    sequences = sorted(os.listdir(sequence))
    labels = sorted(os.listdir(label))
    data = []
    for seq, lab in zip(sequences, labels):
        if seq != lab:
            raise ValueError(f"Sequence and label files do not match for {seq}")
        data.append(convert_sequence(os.path.join(sequence, seq), os.path.join(label, lab), seq))
    return data


def convert_dataset(convert_config):
    dirs = [
        i
        for i in os.listdir(convert_config.dataset_path)
        if os.path.isdir(os.path.join(convert_config.dataset_path, i))
    ]
    splits = sorted({i.split("_")[0] for i in dirs})
    data = {split: convert_pdb_rna_secondary_structure(convert_config.dataset_path, split) for split in splits}
    data["train"] = data.pop("TR1")
    data["validation"] = data.pop("VL1")
    data["test"] = data.pop("TS1")
    data["evaluation"] = data.pop("TS2")
    save_dataset(convert_config, data)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = "pdb-rna_secondary_structure"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
