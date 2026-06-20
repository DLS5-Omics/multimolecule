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

import json
import os
from collections import OrderedDict
from typing import Iterable

import chanfig
import numpy as np
import torch

from multimolecule.models import FactorNetConfig as Config
from multimolecule.models import FactorNetForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream FactorNet (uci-cbcl/FactorNet, Quang & Xie, Methods 2019) one-hot encodes DNA as ["A", "C", "G", "T"];
# the auxiliary per-position signal channels (mappability `Unique35`, DNase cleavage `DGF`, ...) immediately follow
# in the upstream `(L, 4 + num_bws)` input layout.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    print(f"Converting FactorNet checkpoint at {convert_config.checkpoint_path}")
    layout = _load_keras_model(convert_config.checkpoint_path)

    config = Config(
        sequence_length=layout["sequence_length"],
        num_auxiliary_signals=layout["num_auxiliary_signals"],
        num_metadata_features=layout["num_metadata_features"],
        conv_kernel_size=layout["conv_kernel_size"],
        conv_channels=layout["conv_channels"],
        pool_size=layout["pool_size"],
        lstm_hidden_size=layout["lstm_hidden_size"],
        fc_hidden_size=layout["fc_hidden_size"],
        num_labels=layout["num_labels"],
    )
    model = Model(config)

    alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_state_dict(layout, config)

    # Permute the first-layer input channels so the converted weights consume MultiMolecule's DNA token order.
    # The first `vocab_size` channels are the one-hot DNA; the remaining `num_auxiliary_signals` channels are passed
    # through unchanged.
    conv_key = "model.encoder.conv1.weight"
    weight = state_dict[conv_key]
    dna_weight = weight[:, : len(ORIGINAL_VOCAB_LIST), :]
    aux_weight = weight[:, len(ORIGINAL_VOCAB_LIST) :, :]
    dna_weight = convert_one_hot_embeddings(
        dna_weight,
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )
    state_dict[conv_key] = torch.cat([dna_weight, aux_weight], dim=1)

    reference = model.state_dict()
    for key, value in reference.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _load_keras_model(checkpoint_path: str) -> dict:
    """Load an upstream FactorNet trained model directory and return a layout dict + raw Keras tensors.

    The upstream release stores each trained model as a directory containing `model.json` (the Keras 1.1.1
    architecture) plus `best_model.hdf5` (parameter weights), and per-window auxiliary metadata files (`bigwig.txt`,
    `chip.txt`, `meta.txt`). Pass the directory path via `--checkpoint_path`.
    """
    if not (checkpoint_path and os.path.isdir(checkpoint_path)):
        raise FileNotFoundError(
            "Upstream FactorNet checkpoint not found. Clone `uci-cbcl/FactorNet` and point `--checkpoint_path` at "
            "one of its trained model directories (e.g. `models/CTCF/meta_RNAseq_Unique35_DGF`)."
        )
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading the FactorNet Keras HDF5 checkpoint requires the conversion-only dependency `h5py` "
            "(`pip install h5py`)."
        ) from error

    model_json_path = os.path.join(checkpoint_path, "model.json")
    hdf5_path = os.path.join(checkpoint_path, "best_model.hdf5")
    for required in (model_json_path, hdf5_path):
        if not os.path.isfile(required):
            raise FileNotFoundError(f"Missing required upstream file: {required}")
    bigwig_path = os.path.join(checkpoint_path, "bigwig.txt")
    chip_path = os.path.join(checkpoint_path, "chip.txt")
    meta_path = os.path.join(checkpoint_path, "meta.txt")

    with open(model_json_path) as f:
        model_json = json.load(f)
    layers = {layer["name"]: layer for layer in model_json["config"]["layers"]}

    conv = layers["convolution1d_1"]["config"]
    conv_kernel_size = int(conv["filter_length"])
    conv_channels = int(conv["nb_filter"])
    pool = layers["maxpooling1d_1"]["config"]
    pool_size = int(pool["pool_length"])
    if int(pool["stride"]) != pool_size:
        raise ValueError(
            f"FactorNet conversion expects pool stride == pool length, got {pool['stride']} != {pool_size}."
        )
    input_shape = layers["input_1"]["config"]["batch_input_shape"]
    sequence_length = int(input_shape[1])
    total_channels = int(input_shape[2])
    num_auxiliary_signals = total_channels - len(ORIGINAL_VOCAB_LIST)
    if num_auxiliary_signals < 0:
        raise ValueError(
            f"Upstream FactorNet input layer has only {total_channels} channels, fewer than the 4 one-hot DNA "
            "channels; cannot determine auxiliary signal count."
        )

    if "bidirectional_1" in layers:
        lstm_hidden_size = int(layers["bidirectional_1"]["config"]["layer"]["config"]["output_dim"])
    else:
        lstm_hidden_size = 0

    fc_layers = [name for name in layers if layers[name]["class_name"] == "Dense" and name != "timedistributed_1"]
    # `timedistributed_1` is the pointwise dense; the remaining Dense layers are the post-flatten projection(s) and
    # the final sigmoid head. The head's `output_dim` is the number of TFs (== `num_labels`).
    sorted_dense = sorted(fc_layers, key=lambda n: int(n.split("_")[-1]))
    fc_hidden_size = int(layers[sorted_dense[0]]["config"]["output_dim"]) if sorted_dense else conv_channels
    num_labels = int(layers[sorted_dense[-1]]["config"]["output_dim"]) if sorted_dense else 1

    num_metadata_features = 0
    if "input_3" in layers:
        num_metadata_features = int(layers["input_3"]["config"]["batch_input_shape"][1])

    with h5py.File(hdf5_path, "r") as h5:
        # Keras 1.1.x stores per-layer weights under `model_weights/<layer_name>/<weight_dataset_name>`. The dataset
        # name typically has the layer name as prefix (e.g. `convolution1d_1_W`); for `Bidirectional(LSTM)` the
        # `forward_lstm_1_W_i` / `backward_lstm_1_U_c` etc. names are flattened directly inside
        # `model_weights/bidirectional_1/`. For `TimeDistributed(Dense)` the inner dense parameters are flattened
        # too (e.g. `dense_1_W`).
        keras_weights: dict[str, np.ndarray] = {}
        model_weights = h5.get("model_weights", h5)
        for layer_name in model_weights.keys():
            layer_group = model_weights[layer_name]
            if not hasattr(layer_group, "keys"):
                continue
            for dataset_name in layer_group.keys():
                dataset = layer_group[dataset_name]
                if isinstance(dataset, h5py.Dataset):
                    keras_weights[f"{layer_name}/{dataset_name}"] = dataset[()]

    bigwig_names = _read_lines(bigwig_path)
    chip_names = _read_lines(chip_path)
    meta_names = _read_lines(meta_path)

    return {
        "sequence_length": sequence_length,
        "num_auxiliary_signals": num_auxiliary_signals,
        "num_metadata_features": num_metadata_features,
        "conv_kernel_size": conv_kernel_size,
        "conv_channels": conv_channels,
        "pool_size": pool_size,
        "lstm_hidden_size": lstm_hidden_size,
        "fc_hidden_size": fc_hidden_size,
        "num_labels": num_labels,
        "keras_weights": keras_weights,
        "dense_layer_names": sorted_dense,
        "bigwig_names": bigwig_names,
        "chip_names": chip_names,
        "meta_names": meta_names,
    }


def _read_lines(path: str) -> list[str]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _convert_state_dict(layout: dict, config: Config) -> OrderedDict[str, torch.Tensor]:
    keras_weights = layout["keras_weights"]
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    # First Conv1D: Keras 1.1.x `Convolution1D` stores its weight as a 4D tensor of shape
    # `(filter_length, 1, in_channels, nb_filter)` (it dispatches through a 2D conv internally). PyTorch Conv1d
    # expects `(out_channels, in_channels, kernel_size)`.
    conv_weight = _get_keras(keras_weights, "convolution1d_1", ["convolution1d_1_W", "W"])
    conv_bias = _get_keras(keras_weights, "convolution1d_1", ["convolution1d_1_b", "b"])
    if conv_weight.ndim == 4:
        conv_weight = conv_weight[:, 0, :, :]
    state_dict["model.encoder.conv1.weight"] = torch.from_numpy(conv_weight).permute(2, 1, 0).contiguous().float()
    state_dict["model.encoder.conv1.bias"] = torch.from_numpy(conv_bias).contiguous().float()

    # TimeDistributed(Dense): a `(conv_channels, conv_channels)` pointwise projection. Keras 1.1.x stores the inner
    # dense parameters as `dense_1_W` / `dense_1_b` directly under the `timedistributed_1` layer group. Keras stores
    # the weight as `(in_features, out_features)`; PyTorch Linear stores `(out_features, in_features)`.
    pointwise_weight = _get_keras(keras_weights, "timedistributed_1", ["dense_1_W", "W"])
    pointwise_bias = _get_keras(keras_weights, "timedistributed_1", ["dense_1_b", "b"])
    state_dict["model.encoder.pointwise.weight"] = torch.from_numpy(pointwise_weight).t().contiguous().float()
    state_dict["model.encoder.pointwise.bias"] = torch.from_numpy(pointwise_bias).contiguous().float()

    # Bidirectional LSTM (if present): Keras `Bidirectional(LSTM(...))` stores forward and backward LSTM weights
    # under `bidirectional_1/forward_lstm_1/...` and `bidirectional_1/backward_lstm_1/...`. Each LSTM has
    # `W_i, W_f, W_c, W_o` (input projections), `U_i, U_f, U_c, U_o` (recurrent projections), and
    # `b_i, b_f, b_c, b_o` (biases). PyTorch's `nn.LSTM` fuses these into `weight_ih_l0`, `weight_hh_l0`,
    # `bias_ih_l0`, and `bias_hh_l0` with gate order `(i, f, g, o)` (i.e. input / forget / cell / output).
    if config.lstm_hidden_size > 0:
        for direction_name, suffix in (("forward_lstm_1", "l0"), ("backward_lstm_1", "l0_reverse")):
            W_i = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_W_i"])
            W_f = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_W_f"])
            W_c = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_W_c"])
            W_o = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_W_o"])
            U_i = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_U_i"])
            U_f = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_U_f"])
            U_c = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_U_c"])
            U_o = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_U_o"])
            b_i = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_b_i"])
            b_f = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_b_f"])
            b_c = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_b_c"])
            b_o = _get_keras(keras_weights, "bidirectional_1", [f"{direction_name}_b_o"])
            # Keras stores `(in_features, hidden_size)`; PyTorch expects `(4 * hidden_size, in_features)`.
            weight_ih = np.concatenate([W_i, W_f, W_c, W_o], axis=1).T
            weight_hh = np.concatenate([U_i, U_f, U_c, U_o], axis=1).T
            bias = np.concatenate([b_i, b_f, b_c, b_o], axis=0)
            state_dict[f"model.encoder.lstm.weight_ih_{suffix}"] = torch.from_numpy(weight_ih).contiguous().float()
            state_dict[f"model.encoder.lstm.weight_hh_{suffix}"] = torch.from_numpy(weight_hh).contiguous().float()
            # PyTorch additively splits the LSTM bias between `bias_ih` and `bias_hh`; Keras stores a single bias
            # per gate. Put the entire bias on `bias_ih` and zero out `bias_hh`.
            state_dict[f"model.encoder.lstm.bias_ih_{suffix}"] = torch.from_numpy(bias).contiguous().float()
            state_dict[f"model.encoder.lstm.bias_hh_{suffix}"] = torch.zeros_like(
                state_dict[f"model.encoder.lstm.bias_ih_{suffix}"]
            )

    dense_layer_names: list[str] = list(layout["dense_layer_names"])
    if not dense_layer_names:
        raise ValueError("Upstream FactorNet model has no Dense layers; expected at least the final classifier.")

    # First post-flatten Dense -> `model.encoder.dense`.
    flatten_dense = dense_layer_names[0]
    flatten_weight = _get_keras(keras_weights, flatten_dense, [f"{flatten_dense}_W"])
    flatten_bias = _get_keras(keras_weights, flatten_dense, [f"{flatten_dense}_b"])
    state_dict["model.encoder.dense.weight"] = torch.from_numpy(flatten_weight).t().contiguous().float()
    state_dict["model.encoder.dense.bias"] = torch.from_numpy(flatten_bias).contiguous().float()

    # If there is a second pre-head Dense, it is the metadata-fusion projection -> `model.head.dense`. Otherwise
    # the encoder's dense IS the metadata-fusion projection (the upstream `make_model` variant uses a single
    # post-flatten Dense; the upstream `make_meta_model` variant uses two).
    if len(dense_layer_names) >= 3 and config.num_metadata_features > 0:
        meta_dense = dense_layer_names[1]
        meta_weight = _get_keras(keras_weights, meta_dense, [f"{meta_dense}_W"])
        meta_bias = _get_keras(keras_weights, meta_dense, [f"{meta_dense}_b"])
        state_dict["model.head.dense.weight"] = torch.from_numpy(meta_weight).t().contiguous().float()
        state_dict["model.head.dense.bias"] = torch.from_numpy(meta_bias).contiguous().float()
        head_dense = dense_layer_names[-1]
    else:
        # No metadata branch upstream: fuse identity weights into the head's metadata-fusion projection.
        state_dict["model.head.dense.weight"] = torch.eye(config.fc_hidden_size)
        state_dict["model.head.dense.bias"] = torch.zeros(config.fc_hidden_size)
        head_dense = dense_layer_names[-1]
    # Per-TF sigmoid head -> `sequence_head.decoder`.
    head_weight = _get_keras(keras_weights, head_dense, [f"{head_dense}_W"])
    head_bias = _get_keras(keras_weights, head_dense, [f"{head_dense}_b"])
    state_dict["sequence_head.decoder.weight"] = torch.from_numpy(head_weight).t().contiguous().float()
    state_dict["sequence_head.decoder.bias"] = torch.from_numpy(head_bias).contiguous().float()

    return state_dict


def _get_keras(weights: dict, layer: str, suffixes: Iterable[str]) -> np.ndarray:
    """Look up a Keras tensor by `layer/<suffix>`, trying several common Keras 1.1.x naming variants."""
    for suffix in suffixes:
        for key in (
            f"{layer}/{suffix}",
            f"{layer}/{suffix}:0",
            f"{layer}/{layer}/{suffix}",
            f"{layer}/{layer}/{suffix}:0",
        ):
            if key in weights:
                return weights[key]
    raise KeyError(f"Could not locate Keras tensor for layer {layer!r} among suffixes {list(suffixes)!r}.")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
