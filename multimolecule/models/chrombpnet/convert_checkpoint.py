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

import glob
import os
from collections import OrderedDict
from pathlib import Path

import chanfig
import h5py
import torch

from multimolecule.models import ChromBPNetConfig as Config
from multimolecule.models.chrombpnet.modeling_chrombpnet import ChromBPNetForProfilePrediction
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream ChromBPNet (Pampari et al. 2024, kundajelab/chrombpnet) one-hot encodes DNA as ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    print(f"Converting ChromBPNet checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = ChromBPNetForProfilePrediction(config)

    root = convert_config.checkpoint_path
    if root is None or not Path(root).is_dir():
        raise FileNotFoundError(
            "ChromBPNet conversion expects a directory containing *chrombpnet_nobias*.h5 and *bias_scaled*.h5; "
            f"got {root!r}."
        )
    accessibility_ckpt = _find_one(root, "*chrombpnet_nobias*.h5")
    bias_ckpt = _find_one(root, "*bias_scaled*.h5")

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    # Upstream layer naming differs between the two sub-models: the accessibility model uniformly prefixes every
    # layer ("wo_bias_bpnet_..."), while the (scaled) bias model prefixes only its convolutional layers ("bpnet_...")
    # and leaves the profile/count head layers un-prefixed ("prof_out_precrop", "logcount_predictions").
    _convert_branch(
        accessibility_ckpt,
        prefix="model.accessibility",
        conv_prefix="wo_bias_bpnet_",
        head_prefix="wo_bias_bpnet_",
        num_dilated_layers=config.num_dilated_layers,
        num_tasks=config.num_tasks,
        state_dict=state_dict,
    )
    _convert_branch(
        bias_ckpt,
        prefix="model.bias",
        conv_prefix="bpnet_",
        head_prefix="",
        num_dilated_layers=config.bias_num_dilated_layers,
        num_tasks=config.num_tasks,
        state_dict=state_dict,
    )

    for branch in ("accessibility", "bias"):
        key = f"model.{branch}.stem.conv.weight"
        if key not in state_dict:
            raise KeyError(f"Expected stem weight {key} not found in converted state dict.")
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


def _find_one(root: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No ChromBPNet checkpoint matching {pattern!r} found under {root}.")
    return matches[0]


def _read_weight(group: h5py.Group, layer: str, name: str) -> torch.Tensor:
    # Legacy Keras 2.x (.h5) cannot be deserialized by modern Keras, so the raw HDF5 datasets are read directly.
    return torch.from_numpy(group[layer][layer][f"{name}:0"][()])


def _convert_branch(
    file: str,
    prefix: str,
    conv_prefix: str,
    head_prefix: str,
    num_dilated_layers: int,
    num_tasks: int,
    state_dict: OrderedDict,
) -> None:
    with h5py.File(file, "r") as f:
        weights = f["model_weights"]

        # Stem (motif) convolution: Keras Conv1D kernel (kw, in, out) -> torch Conv1d (out, in, kw).
        state_dict[f"{prefix}.stem.conv.weight"] = _read_weight(weights, f"{conv_prefix}1st_conv", "kernel").permute(
            2, 1, 0
        )
        state_dict[f"{prefix}.stem.conv.bias"] = _read_weight(weights, f"{conv_prefix}1st_conv", "bias")

        # Dilated residual convolution stack: <conv_prefix>1conv .. <conv_prefix>(num_dilated_layers)conv.
        for idx in range(num_dilated_layers):
            layer = f"{conv_prefix}{idx + 1}conv"
            state_dict[f"{prefix}.layer.{idx}.conv.weight"] = _read_weight(weights, layer, "kernel").permute(2, 1, 0)
            state_dict[f"{prefix}.layer.{idx}.conv.bias"] = _read_weight(weights, layer, "bias")

        # Profile branch: Keras Conv1D kernel (kw, in, out) -> torch Conv1d (out, in, kw).
        for task_idx in range(num_tasks):
            layer = f"{head_prefix}prof_out_precrop"
            kernel = _read_weight(weights, layer, "kernel").permute(2, 1, 0)
            state_dict[f"{prefix}.profile.{task_idx}.weight"] = kernel
            state_dict[f"{prefix}.profile.{task_idx}.bias"] = _read_weight(weights, layer, "bias")

        # Count branch: Keras Dense kernel (in, out) -> torch Linear (out, in).
        for task_idx in range(num_tasks):
            layer = f"{head_prefix}logcount_predictions"
            state_dict[f"{prefix}.count.{task_idx}.weight"] = _read_weight(weights, layer, "kernel").t()
            state_dict[f"{prefix}.count.{task_idx}.bias"] = _read_weight(weights, layer, "bias")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
