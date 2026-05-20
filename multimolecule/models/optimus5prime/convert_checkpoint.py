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

from multimolecule.models import Optimus5PrimeConfig as Config
from multimolecule.models import Optimus5PrimeForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Optimus 5-Prime one-hot order: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T/U=[0,0,0,1].
# See ``modeling/training_MRL_CNN.ipynb`` in pjsample/human_5utr_modeling. Optimus 5-Prime was trained
# on 5'UTR sequences encoded as DNA (T); MultiMolecule exposes the RNA alphabet (U) so the converter
# remaps the upstream T channel onto MM's U channel.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "U"]

# pjsample/human_5utr_modeling/modeling/saved_models/main_MRL_model.hdf5 -- the headline MRL
# checkpoint highlighted in Figure 2 of the paper.
DEFAULT_CHECKPOINT = "main_MRL_model.hdf5"


def convert_checkpoint(convert_config):
    print(f"Converting Optimus 5-Prime checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    ckpt = convert_config.checkpoint_path
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, DEFAULT_CHECKPOINT)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(config, ckpt)
    key = "encoder.convs.0.weight"
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError(f"Converted state dict is missing {key!r}.")
    converted_weight = convert_one_hot_embeddings(
        weight,
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )
    # Upstream encodes N as an all-zero one-hot row, not as an average nucleotide.
    if "N" in new_vocab_list:
        converted_weight[:, new_vocab_list.index("N"), :] = 0
    state_dict[key] = converted_weight

    # The Keras Flatten layer iterates over ``(sequence_length, conv_filters)`` channels-last; the
    # torch Conv1d output is ``(conv_filters, sequence_length)`` and the encoder transposes back to
    # the upstream channels-last order before flattening. ``dense_1`` therefore consumes the same
    # element order as the original checkpoint without any further permutation here.

    state_dict = OrderedDict(
        (
            key.replace("prediction.", "sequence_head.decoder.") if key.startswith("prediction.") else f"model.{key}",
            value,
        )
        for key, value in state_dict.items()
    )
    model_state = model.state_dict()
    for key in model_state:
        if key not in state_dict and not key.startswith("model.embeddings."):
            raise KeyError(f"Converted state dict is missing the learned parameter {key!r}.")

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config: Config, file: str) -> OrderedDict:
    """Read the legacy Keras (.h5) Optimus 5-Prime weights directly via h5py.

    The upstream checkpoint is a Keras 2 HDF5 file with the layer naming convention
    ``conv1d_{i}`` / ``dense_{i}``. Keras Conv1D kernels are ``(kernel, in_channels, out_channels)``
    and Dense kernels are ``(in, out)``; both are transformed to the torch layout here.
    """
    if not file or not os.path.exists(file):
        raise FileNotFoundError(
            "No upstream Optimus 5-Prime checkpoint found. Download "
            "`main_MRL_model.hdf5` from https://github.com/pjsample/human_5utr_modeling "
            "(``modeling/saved_models/``) and pass it -- or its containing directory -- via "
            "`--checkpoint_path`."
        )
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as f:
        weights = f["model_weights"]
        for index in range(config.num_conv_layers):
            keras_name = f"conv1d_{index + 1}"
            group = weights[keras_name][keras_name]
            kernel = torch.from_numpy(group["kernel:0"][()]).float()
            bias = torch.from_numpy(group["bias:0"][()]).float()
            # Keras Conv1D kernel: (kernel, in_channels, out_channels) -> torch (out, in, kernel).
            kernel = kernel.permute(2, 1, 0).contiguous()
            state_dict[f"encoder.convs.{index}.weight"] = kernel
            state_dict[f"encoder.convs.{index}.bias"] = bias

        dense1 = weights["dense_1"]["dense_1"]
        state_dict["encoder.dense.weight"] = torch.from_numpy(dense1["kernel:0"][()]).float().t().contiguous()
        state_dict["encoder.dense.bias"] = torch.from_numpy(dense1["bias:0"][()]).float()

        dense2 = weights["dense_2"]["dense_2"]
        state_dict["prediction.weight"] = torch.from_numpy(dense2["kernel:0"][()]).float().t().contiguous()
        state_dict["prediction.bias"] = torch.from_numpy(dense2["bias:0"][()]).float()
    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = os.path.join(os.path.dirname(__file__), "saved_models")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
