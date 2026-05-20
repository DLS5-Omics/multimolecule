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

from multimolecule.models import MpraDragoNnConfig as Config
from multimolecule.models import MpraDragoNnForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream MPRA-DragoNN (kundajelab/MPRA-DragoNN ConvModel) one-hot encodes DNA in the order
# ["A", "C", "G", "T"] (see kipoi/ConvModel/model.yaml).
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Mapping from upstream Keras layer names to MultiMolecule module prefixes. The released
# `kipoi/ConvModel/pretrained.hdf5` (https://github.com/kundajelab/mpra_minimal/raw/87197541b/
# kipoi/ConvModel/pretrained.hdf5) is a Keras 2.2.4 Sequential model with this layer order:
#   conv1d_1 (relu, valid) -> batch_normalization_1 -> dropout_1
#   conv1d_2 (relu, valid) -> batch_normalization_2 -> dropout_2
#   conv1d_3 (relu, valid) -> batch_normalization_3 -> dropout_3
#   flatten_1 -> dense_1 (12 units, linear)
CONV_LAYERS = ["conv1d_1", "conv1d_2", "conv1d_3"]
ENCODER_NORMS = ["batch_normalization_1", "batch_normalization_2", "batch_normalization_3"]
HEAD_LAYER = "dense_1"


def convert_checkpoint(convert_config):
    print(f"Converting MPRA-DragoNN checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.input_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    # Upstream Keras flattens `(length, channels)` (channels-last) while MultiMolecule's `nn.Flatten`
    # operates on `(channels, length)` (channels-first after Conv1d). The first dense layer's input
    # weight is reindexed to bridge the two flattening orders.
    last_channels = config.conv_channels[-1]
    pooled_length = config.pooled_length

    root = convert_config.checkpoint_path
    if root and os.path.isfile(root):
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
        # The upstream `pretrained.hdf5` is hosted at
        # https://github.com/kundajelab/mpra_minimal/raw/87197541b/kipoi/ConvModel/pretrained.hdf5
        # (MIT licensed; md5 19fb17f943c3d6bcada8c5dc638092b4).
        raise FileNotFoundError(
            "No upstream MPRA-DragoNN checkpoint found. Download "
            "`kipoi/ConvModel/pretrained.hdf5` from "
            "https://github.com/kundajelab/MPRA-DragoNN/raw/master/kipoi/ConvModel/pretrained.hdf5 "
            "(or the equivalent kundajelab/mpra_minimal raw URL) and pass it via `--checkpoint_path`."
        )

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _reorder_flatten_dense(state_dict: OrderedDict, channels: int, length: int) -> OrderedDict:
    """Bridge the Keras `(length, channels)` flatten order to the torch `(channels, length)` order.

    Keras flattens the final feature map (shape `(length, channels)`) so the input feature at
    Keras column ``l * channels + c`` corresponds to the torch column ``c * length + l``. The
    decoder weight (shape `(out, length * channels)`) is reindexed accordingly so the torch model
    reproduces the upstream computation without any architecture change.
    """
    key = "sequence_head.decoder.weight"
    weight = state_dict.get(key)
    if weight is None:
        return state_dict
    in_features = weight.shape[1]
    if in_features != channels * length:
        raise ValueError(
            f"Unexpected decoder input size {in_features}; expected channels*length = {channels * length}."
        )
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

        head_weights = _read_layer_weights(h5file, HEAD_LAYER)
        # Keras Dense kernel: (in_features, out_features) -> torch (out, in)
        decoder_weight = torch.from_numpy(head_weights["kernel"]).transpose(0, 1).contiguous()
        state_dict["sequence_head.decoder.weight"] = decoder_weight
        state_dict["sequence_head.decoder.bias"] = torch.from_numpy(head_weights["bias"])

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
