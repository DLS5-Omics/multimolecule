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

from multimolecule.models import XpressoConfig as Config
from multimolecule.models import XpressoForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Xpresso (vagarwal87/Xpresso) one-hot encodes the promoter sequence ordered as ["A", "C", "G", "T"]
# (see `one_hot()` in upstream `xpresso_predict.py`: ``seqindex = {'A':0,'C':1,'G':2,'T':3}``).
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Map each upstream Keras layer to its MultiMolecule state-dict prefix. The pretrained graph is:
#   promoter -> conv1d_1 -> relu -> max_pooling1d_1
#            -> conv1d_2 -> relu -> max_pooling1d_2 -> flatten_1
#   [flatten_1, halflife] -> concatenate_1
#            -> dense_1 -> relu -> dropout_1
#            -> dense_2 -> relu -> dropout_2
#            -> dense_3 (linear, scalar output)
# `dense_3` is the unactivated scalar regression layer and maps onto the shared
# `SequencePredictionHead.decoder` carried by `XpressoForSequencePrediction`.
LAYER_MAPPING = {
    "conv1d_1": "model.encoder.blocks.0.conv",
    "conv1d_2": "model.encoder.blocks.1.conv",
    "dense_1": "model.head.layers.0.dense",
    "dense_2": "model.head.layers.1.dense",
    "dense_3": "sequence_head.decoder",
}


def convert_checkpoint(convert_config):
    print(f"Converting Xpresso checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.input_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config, config)
    key = "model.encoder.blocks.0.conv.weight"
    weight = state_dict.get(key)
    if weight is not None:
        state_dict[key] = convert_one_hot_embeddings(
            weight,
            old_vocab=ORIGINAL_VOCAB_LIST,
            new_vocab=new_vocab_list,
            convert_word_embeddings=convert_word_embeddings,
        )
    load_checkpoint(model, state_dict)

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def convert_original_state_dict_value(name: str, value: torch.Tensor) -> torch.Tensor:
    """Convert Keras tensor layouts to torch.

    - Keras `Conv1D` kernels are `(kernel_size, in_channels, out_channels)`; torch expects
      `(out_channels, in_channels, kernel_size)` -> permute `(2, 1, 0)`.
    - Keras `Dense` kernels are `(in_features, out_features)`; torch `nn.Linear.weight` is
      `(out_features, in_features)` -> transpose.
    """
    if "conv" in name and value.ndim == 3:
        return value.permute(2, 1, 0).contiguous()
    if ("dense" in name or "decoder" in name) and value.ndim == 2:
        return value.t().contiguous()
    return value


def _flatten_permutation(config: Config) -> torch.Tensor:
    """Index map from MM's `torch.flatten` ordering to upstream Keras `Flatten` ordering.

    The final convolutional feature map has `channels` channels and `length` positions. Upstream
    Keras keeps it as `(length, channels)` and `Flatten()` walks it row-major, so element
    `(l, c)` lands at flat index ``l * channels + c``. MultiMolecule keeps the tensor as
    `(channels, length)` and `torch.flatten` produces ``c * length + l``.

    `dense_1` was trained against the Keras ordering, so the converted weight's input rows must be
    reindexed from the Keras ordering into the MultiMolecule ordering before loading. The returned
    tensor `perm` satisfies ``mm_input[i] == keras_input[perm[i]]``; selecting columns of the
    transposed Keras kernel with `perm` yields a torch weight that consumes the MM-ordered vector.
    """
    length = config.input_length
    channels = config.vocab_size
    for index in range(config.num_conv_layers):
        channels = config.conv_channels[index]
        length = length // config.pool_sizes[index]
    # MM flat index i = c * length + l  ->  Keras flat index = l * channels + c
    c = torch.arange(channels).repeat_interleave(length)
    length_index = torch.arange(length).repeat(channels)
    return length_index * channels + c


def _reorder_dense1_inputs(state_dict: OrderedDict, config: Config) -> OrderedDict:
    """Permute `dense_1` input columns so MM's flatten ordering matches the trained Keras ordering.

    `dense_1` consumes ``concatenate([flatten(conv), halflife])``. Only the convolutional block
    (the first `flattened` columns) is affected by the flatten-ordering mismatch; the auxiliary
    half-life features are appended identically by both frameworks and stay in place.
    """
    key = "model.head.layers.0.dense.weight"
    weight = state_dict.get(key)
    if weight is None:
        return state_dict
    length = config.input_length
    channels = config.vocab_size
    for index in range(config.num_conv_layers):
        channels = config.conv_channels[index]
        length = length // config.pool_sizes[index]
    flattened = channels * length
    perm = _flatten_permutation(config)
    new_weight = weight.clone()
    new_weight[:, :flattened] = weight[:, :flattened][:, perm]
    state_dict[key] = new_weight
    return state_dict


def _convert_checkpoint(convert_config, config: Config) -> OrderedDict:
    """Load and convert an upstream Keras Xpresso checkpoint.

    The upstream release ships HDF5 (`.h5`) Keras model files (e.g.
    `humanMedian_trainepoch.11-0.426.h5`, distributed inside `Xpresso-predict.zip`). They were
    saved with Keras 2.0.5, whose serialization is not loadable by modern Keras. The weights are
    therefore read straight out of the HDF5 `model_weights` group with `h5py` (no Keras runtime
    required), names are translated via `LAYER_MAPPING`, and tensor layouts are converted via
    `convert_original_state_dict_value`.
    """
    checkpoint_path = convert_config.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "No upstream Xpresso checkpoint found. Download `Xpresso-predict.zip` from the official "
            "Xpresso data release, unzip it, and pass a `.h5` checkpoint file or a directory containing one "
            "via `--checkpoint_path`."
        )
    if os.path.isdir(checkpoint_path):
        ckpts = sorted(f for f in os.listdir(checkpoint_path) if f.endswith(".h5"))
        if not ckpts:
            raise FileNotFoundError(f"No `.h5` Xpresso checkpoint found in directory {checkpoint_path!r}.")
        checkpoint_path = os.path.join(checkpoint_path, ckpts[0])

    import h5py  # noqa: PLC0415  isolate optional conversion-only dependency

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(checkpoint_path, "r") as h5:
        weights = h5["model_weights"]
        for keras_name, mm_prefix in LAYER_MAPPING.items():
            group = weights[keras_name]
            weight_names = group.attrs["weight_names"]
            for raw in weight_names:
                wname = raw.decode() if isinstance(raw, bytes) else raw
                node = group
                for part in wname.split("/"):
                    node = node[part]
                array = node[()]
                # ``kernel:0`` -> weight, ``bias:0`` -> bias.
                leaf = wname.split("/")[-1].split(":")[0]
                suffix = "weight" if leaf == "kernel" else "bias"
                key = f"{mm_prefix}.{suffix}"
                tensor = convert_original_state_dict_value(key, torch.from_numpy(array))
                state_dict[key] = tensor

    state_dict = _reorder_dense1_inputs(state_dict, config)
    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    # Upstream models live at https://github.com/vagarwal87/Xpresso. The pretrained Keras `.h5`
    # files ship inside `Xpresso-predict.zip`:
    #   https://krishna.gs.washington.edu/content/members/vagar/Xpresso/data/Xpresso-predict.zip
    # (mirror of the 403-gated https://xpresso.gs.washington.edu/data/Xpresso-predict.zip). Unzip
    # and place `pretrained_models/humanMedian_trainepoch.11-0.426.h5` (or another `.h5`) under
    # `checkpoint_path` before running.
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
