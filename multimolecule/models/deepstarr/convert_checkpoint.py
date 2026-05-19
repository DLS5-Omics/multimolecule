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

from multimolecule.models import DeepStarrConfig as Config
from multimolecule.models import DeepStarrForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream DeepSTARR (bernardo-de-almeida/DeepSTARR) one-hot encodes DNA in the order ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Mapping from upstream Keras layer names to MultiMolecule module prefixes.
# The upstream DeepSTARR model (Zenodo record 5502060, DeepSTARR.model.h5) is a Keras 2.2.4
# functional model with the following layers:
#   Conv1D_1st / Conv1D_2 / Conv1D_3 / Conv1D_4          -> encoder convolutions
#   batch_normalization_60..63                           -> encoder batch norms
#   Dense_1 / Dense_2                                    -> fully connected pooler
#   batch_normalization_64 / batch_normalization_65      -> pooler batch norms
#   Dense_Dev / Dense_Hk (each 1 output unit)            -> the two regression heads
CONV_LAYERS = ["Conv1D_1st", "Conv1D_2", "Conv1D_3", "Conv1D_4"]
ENCODER_NORMS = [
    "batch_normalization_60",
    "batch_normalization_61",
    "batch_normalization_62",
    "batch_normalization_63",
]
DENSE_LAYERS = ["Dense_1", "Dense_2"]
POOLER_NORMS = ["batch_normalization_64", "batch_normalization_65"]
HEAD_LAYERS = ["Dense_Dev", "Dense_Hk"]


def convert_checkpoint(convert_config):
    print(f"Converting DeepStarr checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.input_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    # The upstream Keras `Flatten` operates on a `(length, channels)` feature map (length-major
    # order), while MultiMolecule's `nn.Flatten` operates on `(channels, length)` (channel-major
    # order). The first pooler `Dense` weight is reindexed to bridge the two flattening orders.
    last_channels = config.conv_channels[-1]
    pooled_length = config.input_length
    for _ in range(config.num_conv_layers):
        pooled_length //= config.pool_size

    root = convert_config.checkpoint_path
    if root is not None and os.path.isfile(root):
        state_dict = _convert_checkpoint(root)
        state_dict = _reorder_flatten_dense(state_dict, last_channels, pooled_length)
        key = "model.encoder.blocks.0.conv.weight"
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
    else:
        raise FileNotFoundError(
            "No upstream DeepSTARR checkpoint found. Download `DeepSTARR.model.h5` from Zenodo record 5502060 "
            "(https://zenodo.org/records/5502060) and pass it via `--checkpoint_path`."
        )

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _reorder_flatten_dense(state_dict: OrderedDict, channels: int, length: int) -> OrderedDict:
    """Bridge the Keras `(length, channels)` flatten order to the torch `(channels, length)` order.

    Keras flattens the last pooled feature map (shape `(length, channels)`) so the input feature at
    Keras column ``l * channels + c`` corresponds to the torch column ``c * length + l``. The first
    pooler `Dense` weight (shape `(out, length * channels)`) is reindexed accordingly so the
    torch-only model reproduces the upstream computation without any architecture change.
    """
    key = "model.pooler.layers.0.dense.weight"
    weight = state_dict.get(key)
    if weight is None:
        return state_dict
    in_features = weight.shape[1]
    if in_features != channels * length:
        raise ValueError(f"Unexpected pooler input size {in_features}; expected channels*length = {channels * length}.")
    # torch flatten enumerates columns in (channel, length) order: torch column j -> c = j // length,
    # l = j % length. The matching Keras column (length, channels order) is l * channels + c.
    torch_columns = torch.arange(in_features)
    c = torch_columns // length
    length_index = torch_columns % length
    keras_source = length_index * channels + c
    state_dict[key] = weight.index_select(1, keras_source).contiguous()
    return state_dict


def _read_layer_weights(h5file, layer_name: str) -> dict:
    """Read the weight tensors of a single Keras layer from an HDF5 weights file.

    The upstream checkpoint was saved by Keras 2.2.4, so weights live under
    `model_weights/<layer>/<layer>/<weight>:0` (or `<layer>/<inner>/<weight>:0`).
    """
    import numpy as np  # noqa: PLC0415

    group = h5file["model_weights"][layer_name] if "model_weights" in h5file else h5file[layer_name]
    weights: dict[str, np.ndarray] = {}

    def _collect(node, prefix=""):
        import h5py  # noqa: PLC0415

        for key in node.keys():
            item = node[key]
            if isinstance(item, h5py.Group):
                _collect(item, prefix)
            else:
                short = key.split(":")[0]
                weights[short] = item[()]

    _collect(group)
    return weights


def _convert_checkpoint(file) -> OrderedDict:
    import h5py  # noqa: PLC0415

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as h5file:
        # Convolutional encoder blocks.
        for index, (conv, norm) in enumerate(zip(CONV_LAYERS, ENCODER_NORMS)):
            conv_weights = _read_layer_weights(h5file, conv)
            # Keras Conv1D kernel: (kernel, in_channels, out_channels) -> torch (out, in, kernel)
            kernel = torch.from_numpy(conv_weights["kernel"]).permute(2, 1, 0).contiguous()
            state_dict[f"model.encoder.blocks.{index}.conv.weight"] = kernel
            state_dict[f"model.encoder.blocks.{index}.conv.bias"] = torch.from_numpy(conv_weights["bias"])
            _convert_batchnorm(
                state_dict,
                _read_layer_weights(h5file, norm),
                f"model.encoder.blocks.{index}.norm",
            )

        # Fully connected pooler layers.
        for index, (dense, norm) in enumerate(zip(DENSE_LAYERS, POOLER_NORMS)):
            dense_weights = _read_layer_weights(h5file, dense)
            # Keras Dense kernel: (in_features, out_features) -> torch (out, in)
            kernel = torch.from_numpy(dense_weights["kernel"]).transpose(0, 1).contiguous()
            state_dict[f"model.pooler.layers.{index}.dense.weight"] = kernel
            state_dict[f"model.pooler.layers.{index}.dense.bias"] = torch.from_numpy(dense_weights["bias"])
            _convert_batchnorm(
                state_dict,
                _read_layer_weights(h5file, norm),
                f"model.pooler.layers.{index}.norm",
            )

        # The two single-unit regression heads (Dense_Dev, Dense_Hk) are concatenated into the
        # MultiMolecule sequence prediction head's decoder (2 outputs).
        head_kernels, head_biases = [], []
        for head in HEAD_LAYERS:
            head_weights = _read_layer_weights(h5file, head)
            head_kernels.append(torch.from_numpy(head_weights["kernel"]).transpose(0, 1).contiguous())
            head_biases.append(torch.from_numpy(head_weights["bias"]))
        state_dict["sequence_head.decoder.weight"] = torch.cat(head_kernels, dim=0).contiguous()
        state_dict["sequence_head.decoder.bias"] = torch.cat(head_biases, dim=0).contiguous()

    return state_dict


def _convert_batchnorm(state_dict: OrderedDict, weights: dict, prefix: str) -> None:
    state_dict[f"{prefix}.weight"] = torch.from_numpy(weights["gamma"])
    state_dict[f"{prefix}.bias"] = torch.from_numpy(weights["beta"])
    state_dict[f"{prefix}.running_mean"] = torch.from_numpy(weights["moving_mean"])
    state_dict[f"{prefix}.running_var"] = torch.from_numpy(weights["moving_variance"])


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
