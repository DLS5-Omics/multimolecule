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
from collections import OrderedDict

import chanfig
import keras
import torch
from keras.models import load_model

from multimolecule.models import SpliceAiConfig as Config
from multimolecule.models import SpliceAiModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import get_alphabet, get_tokenizer_config

try:
    from packaging.version import parse as parse_version
except ImportError:
    from pkg_resources import parse_version  # type: ignore

keras_version = parse_version(keras.__version__)

torch.manual_seed(1016)


name_mapping = {
    "conv1d": "conv",
    "batch_normalization": "norm",
    "kernel": "weight",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    "gamma": "weight",
    "beta": "bias",
}


def _convert_name(name: str) -> str:
    name = name.replace("/", ".")[:-2]

    for old, new in name_mapping.items():
        name = name.replace(old, new)

    counter = 3 if "running" in name else 2
    if ("conv" in name or "norm" in name) and name.count("_") == counter:
        prefix, suffix = name.split(".", 1)
        name = ".".join([prefix[:-2], suffix])

    if "conv" in name:
        prefix, suffix = name.split(".", 1)
        layer_type, layer_idx_str = prefix.rsplit("_", 1)
        layer_idx = int(layer_idx_str)

        if layer_idx % 9 == 2 and layer_idx > 9:
            group_idx = layer_idx // 9 - 1
            name = f"encoder.stages.{group_idx}.{layer_type}.{suffix}"
        elif layer_idx == 1:
            name = f"embedding.{suffix}"
        elif layer_idx == 2:
            name = f"encoder.conv.{suffix}"
        elif layer_idx == 39:
            name = f"prediction.{suffix}"
        else:
            adjusted_idx = layer_idx - (layer_idx - 2) // 9 - 2
            name = f"{layer_type}_{adjusted_idx}.{suffix}"

    if "conv_" in name or "norm_" in name:
        try:
            prefix, suffix = name.split(".", 1)
            layer_type, layer_idx_str = prefix.rsplit("_", 1)
            layer_idx = int(layer_idx_str)

            block_idx = (layer_idx - 1) // 2 + 1
            layer_position = layer_idx % 2

            if layer_position == 0:
                layer_position = 2

            group_idx = (block_idx - 1) // 4
            block_position = block_idx % 4 - 1

            if block_position == -1:
                block_position = 3

            name = f"encoder.stages.{group_idx}.blocks.{block_position}.{layer_type}_{layer_position}.{suffix}"

            name = name.replace("conv_", "conv")
            name = name.replace("norm_", "norm")
        except ValueError:
            pass

    return name


def _convert_checkpoint(file):
    model = load_model(file)
    state_dict = OrderedDict()
    for layer in model.layers:
        weight_names = [w.name for w in layer.weights]
        if keras_version > parse_version("3.0"):
            weight_names = [layer.name + "/" + n + "00" for n in weight_names]
        weight_values = layer.get_weights()
        for name, value in zip(weight_names, weight_values):
            new_name = _convert_name(name)
            new_value = torch.from_numpy(value)
            if "kernel" in name:
                new_value = new_value.transpose(0, 2)
            state_dict[new_name] = new_value
    return state_dict


def convert_checkpoint(convert_config):
    config = Config()
    config.architectures = ["SpliceAiModel"]

    model = Model(config)

    root = convert_config.checkpoint_path
    ckpts = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".h5")])
    for i, ckpt in enumerate(ckpts):
        state_dict = _convert_checkpoint(ckpt)
        model.networks[i].load_state_dict(state_dict)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config["unk_token"] = "N"
    tokenizer_config["pad_token"] = "N"
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
