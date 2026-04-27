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

import chanfig
import torch

from multimolecule.models import UfoldConfig as Config
from multimolecule.models import UfoldModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import get_alphabet, get_tokenizer_config


def convert_checkpoint(convert_config) -> None:
    print(f"Converting UFold checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    config.architectures = ["UfoldModel"]
    model = Model(config)

    checkpoint = torch.load(convert_config.checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state dict checkpoint, but got {type(checkpoint)}.")

    vocab_list = get_alphabet("nucleobase", prepend_tokens=[]).vocabulary
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        key = f"encoder.{convert_original_state_dict_key(key)}"
        if key == "encoder.down_blocks.0.conv1.weight":
            pairwise_channels = [
                original_vocab_list.index(i) * len(original_vocab_list) + original_vocab_list.index(j)
                for i in vocab_list
                for j in vocab_list
            ]
            channels = torch.tensor([*pairwise_channels, len(pairwise_channels)], device=value.device)
            value = value.index_select(1, channels)
        state_dict[key] = value
    state_dict["criterion.pos_weight"] = model.criterion.pos_weight

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def convert_original_state_dict_key(key: str) -> str:
    for block_index in range(1, 6):
        prefix = f"Conv{block_index}.conv."
        if key.startswith(prefix):
            return f"down_blocks.{block_index - 1}.{convert_original_conv_block_key(key.removeprefix(prefix))}"

    for block_index, original_index in enumerate(range(5, 1, -1)):
        prefix = f"Up{original_index}.up."
        if key.startswith(prefix):
            return f"up_blocks.{block_index}.{convert_original_up_block_key(key.removeprefix(prefix))}"

        prefix = f"Up_conv{original_index}.conv."
        if key.startswith(prefix):
            return f"decoder_blocks.{block_index}.{convert_original_conv_block_key(key.removeprefix(prefix))}"

    if key.startswith("Conv_1x1."):
        return f"prediction.{key.removeprefix('Conv_1x1.')}"
    return key


def convert_original_conv_block_key(key: str) -> str:
    replacements = {
        "0.": "conv1.",
        "1.": "batch_norm1.",
        "3.": "conv2.",
        "4.": "batch_norm2.",
    }
    for source, target in replacements.items():
        if key.startswith(source):
            return f"{target}{key.removeprefix(source)}"
    return key


def convert_original_up_block_key(key: str) -> str:
    replacements = {
        "1.": "conv.",
        "2.": "batch_norm.",
    }
    for source, target in replacements.items():
        if key.startswith(source):
            return f"{target}{key.removeprefix(source)}"
    return key


original_vocab_list = ["A", "U", "C", "G"]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
