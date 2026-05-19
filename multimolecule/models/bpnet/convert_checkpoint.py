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
import h5py
import torch

from multimolecule.models import BpNetConfig as Config
from multimolecule.models.bpnet.modeling_bpnet import BpNetForProfilePrediction
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream BPNet (Avsec et al. 2021, kundajelab/bpnet, BPNet-OSKN) one-hot encodes DNA as ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Order of the four prediction tasks in the published BPNet-OSKN checkpoint.
TASKS = ["Oct4", "Sox2", "Nanog", "Klf4"]


def convert_checkpoint(convert_config):
    print(f"Converting BPNet checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = BpNetForProfilePrediction(config)

    root = convert_config.checkpoint_path
    if not root:
        raise FileNotFoundError(
            "No upstream BPNet checkpoint found. Download `bpnet.model.h5` from the BPNet-OSKN Zenodo "
            "artifact and pass the file or containing directory via `--checkpoint_path`."
        )
    ckpt = root if os.path.isfile(root) else os.path.join(root, "bpnet.model.h5")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            "No upstream BPNet checkpoint found. Download `bpnet.model.h5` from the BPNet-OSKN Zenodo "
            "artifact and pass the file or containing directory via `--checkpoint_path`."
        )

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(ckpt, config)
    key = "model.encoder.stem.conv.weight"
    state_dict[key] = convert_one_hot_embeddings(
        state_dict[key],
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    reference = model.state_dict()
    for key, value in reference.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _read_weight(group: h5py.Group, layer: str, name: str) -> torch.Tensor:
    # Legacy Keras 2.2.4 (.h5) cannot be deserialized by modern Keras, so the raw HDF5 datasets are read directly.
    return torch.from_numpy(group[layer][layer][f"{name}:0"][()])


def _convert_checkpoint(file: str, config: Config) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as f:
        weights = f["model_weights"]

        # Stem (motif) convolution: Keras Conv1D kernel (kw, in, out) -> torch Conv1d (out, in, kw).
        state_dict["model.encoder.stem.conv.weight"] = _read_weight(weights, "conv1d_1", "kernel").permute(2, 1, 0)
        state_dict["model.encoder.stem.conv.bias"] = _read_weight(weights, "conv1d_1", "bias")

        # Dilated residual convolution stack: conv1d_2 .. conv1d_(num_dilated_layers + 1).
        for idx in range(config.num_dilated_layers):
            layer = f"conv1d_{idx + 2}"
            state_dict[f"model.encoder.layer.{idx}.conv.weight"] = _read_weight(weights, layer, "kernel").permute(
                2, 1, 0
            )
            state_dict[f"model.encoder.layer.{idx}.conv.bias"] = _read_weight(weights, layer, "bias")

        # Profile branch: Keras Conv2DTranspose kernel (kh, 1, out, in) -> torch ConvTranspose1d (in, out, kh).
        for task_idx in range(config.num_tasks):
            layer = f"conv2d_transpose_{task_idx + 1}"
            kernel = _read_weight(weights, layer, "kernel").squeeze(1).permute(2, 1, 0)
            state_dict[f"profile_count_head.profile.{task_idx}.weight"] = kernel
            state_dict[f"profile_count_head.profile.{task_idx}.bias"] = _read_weight(weights, layer, "bias")

        # Count branch: Keras Dense kernel (in, out) -> torch Linear (out, in).
        for task_idx in range(config.num_tasks):
            layer = f"dense_{2 * task_idx + 1}"
            state_dict[f"profile_count_head.count.{task_idx}.weight"] = _read_weight(weights, layer, "kernel").t()
            state_dict[f"profile_count_head.count.{task_idx}.bias"] = _read_weight(weights, layer, "bias")

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
