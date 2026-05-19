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
import torch

from multimolecule.models import BasenjiConfig as Config
from multimolecule.models import BasenjiForTokenPrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream calico/basenji one-hot encodes DNA in the order ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Upstream Basenji2 (calico/basenji `manuscripts/cross2020/params_human.json`) ships a flat Keras
# graph whose layers are named `conv1d`, `conv1d_1`, ..., `batch_normalization`, ..., `dense`.
# The convolutions/batch-norms appear in execution order, so a single linear index maps each one
# onto the hierarchical MultiMolecule module tree:
#   conv index 0          -> encoder.stem
#   conv index 1..6       -> encoder.conv_tower.{0..5}
#   conv index 7..28      -> encoder.blocks.{k} (7+2k -> conv1, 8+2k -> conv2)
#   conv index 29         -> encoder.head
#   dense                 -> token_head.decoder
NUM_TOWER_STAGES = 6
NUM_DILATED_BLOCKS = 11


def convert_checkpoint(convert_config):
    print(f"Converting Basenji checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path)
    key = "model.encoder.stem.conv1.weight"
    weight = state_dict.get(key)
    if weight is not None:
        state_dict[key] = convert_one_hot_embeddings(
            weight,
            old_vocab=ORIGINAL_VOCAB_LIST,
            new_vocab=new_vocab_list,
            convert_word_embeddings=convert_word_embeddings,
        )

    reference_state = model.state_dict()
    for key, value in reference_state.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _conv_destination(index: int) -> str:
    """Map an upstream conv/batch-norm execution index to its MultiMolecule module prefix."""
    if index == 0:
        return "encoder.stem"
    if 1 <= index <= NUM_TOWER_STAGES:
        return f"encoder.conv_tower.{index - 1}"
    if index == 1 + NUM_TOWER_STAGES + 2 * NUM_DILATED_BLOCKS:
        return "encoder.head"
    block = (index - (1 + NUM_TOWER_STAGES)) // 2
    sub = "conv1" if (index - (1 + NUM_TOWER_STAGES)) % 2 == 0 else "conv2"
    return f"encoder.blocks.{block}.{sub}"


def _convert_checkpoint(file) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    if not file or not os.path.exists(file):
        raise FileNotFoundError(
            "No upstream Basenji checkpoint found. Download `model_human.h5` from the Basenji "
            "cross-species release (`https://storage.googleapis.com/basenji_barnyard2/model_human.h5`) "
            "and pass it via `--checkpoint_path`."
        )

    import h5py  # noqa: PLC0415  # transient conversion-only dependency, never imported at runtime

    with h5py.File(file, "r") as h5:
        weights = h5["model_weights"]

        def read(layer: str, var: str) -> torch.Tensor:
            return torch.from_numpy(weights[layer][layer][f"{var}:0"][()])

        n_conv = 1 + NUM_TOWER_STAGES + 2 * NUM_DILATED_BLOCKS + 1
        for index in range(n_conv):
            layer = "conv1d" if index == 0 else f"conv1d_{index}"
            bn = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
            prefix = _conv_destination(index)
            # Keras Conv1D kernels are (kernel_size, in_channels, out_channels);
            # torch Conv1d expects (out_channels, in_channels, kernel_size).
            state_dict[f"model.{prefix}.conv1.weight"] = read(layer, "kernel").permute(2, 1, 0).contiguous()
            state_dict[f"model.{prefix}.batch_norm1.weight"] = read(bn, "gamma")
            state_dict[f"model.{prefix}.batch_norm1.bias"] = read(bn, "beta")
            state_dict[f"model.{prefix}.batch_norm1.running_mean"] = read(bn, "moving_mean")
            state_dict[f"model.{prefix}.batch_norm1.running_var"] = read(bn, "moving_variance")

        # Final `Dense (head_hidden_size -> num_labels)`: Keras stores (in, out); torch wants
        # (out, in). The softplus activation is applied by the model, not stored as a weight.
        state_dict["token_head.decoder.weight"] = read("dense", "kernel").t().contiguous()
        state_dict["token_head.decoder.bias"] = read("dense", "bias")

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
