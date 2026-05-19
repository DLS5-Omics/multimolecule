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

from multimolecule.models import AparentConfig as Config
from multimolecule.models import AparentForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream APARENT one-hot order is ["A", "C", "G", "T"] (aparent/predictor/aparent_predictor.py).
# MultiMolecule exposes 3'UTR/polyA sequence as RNA, so the upstream T channel is mapped to U.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "U"]

# APARENT distributes several Keras checkpoints; the base, non-normalised model recommended
# upstream for isoform / variant-effect prediction is the one converted here.
DEFAULT_CHECKPOINT = "aparent_large_lessdropout_all_libs_no_sampleweights.h5"


def convert_checkpoint(convert_config):
    print(f"Converting Aparent checkpoint at {convert_config.checkpoint_path}")
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

    state_dict = _convert_checkpoint(ckpt)
    key = "encoder.conv1.weight"
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError("Converted state dict is missing 'encoder.conv1.weight'.")
    state_dict[key] = convert_one_hot_embeddings(
        weight,
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    model_state = model.model.state_dict()
    for key in model_state:
        if key not in state_dict and not key.startswith(("embeddings.", "encoder.distal_pas", "encoder.library")):
            raise KeyError(f"Converted state dict is missing the learned parameter {key!r}.")
    state_dict = OrderedDict((f"model.{key}", value) for key, value in state_dict.items())

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


# Maps the upstream Keras layer name to the MultiMolecule parameter prefix.
NAME_MAPPING = {
    "conv2d_1": "encoder.conv1",
    "conv2d_2": "encoder.conv2",
    "dense_1": "encoder.dense1",
    "dense_2": "encoder.dense2",
    "dense_3": "cleavage_decoder",
    "dense_4": "isoform_decoder",
}


def _convert_checkpoint(file: str) -> OrderedDict:
    """Read the legacy Keras (.h5) weights directly via h5py.

    The upstream checkpoint is a Keras 2.1.3 HDF5 file; modern Keras cannot deserialise the
    legacy graph, so the raw weight datasets are read instead. Keras stores Conv kernels as
    ``(kernel, ..., in, out)`` and Dense kernels as ``(in, out)``; both are transformed to
    the torch layout here.
    """
    if not file or not os.path.exists(file):
        raise FileNotFoundError(
            "No upstream APARENT checkpoint found. Download the APARENT saved models from the official "
            "repository and pass `aparent_large_lessdropout_all_libs_no_sampleweights.h5` or its containing "
            "directory via `--checkpoint_path`."
        )
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as f:
        weights = f["model_weights"]
        for keras_name, torch_prefix in NAME_MAPPING.items():
            group = weights[keras_name][keras_name]
            kernel = torch.from_numpy(group["kernel:0"][()]).float()
            bias = torch.from_numpy(group["bias:0"][()]).float()
            if keras_name.startswith("conv2d"):
                # Keras Conv2D kernel: (kernel, nucleotide_or_1, in_channels, out_channels).
                # APARENT's convs span the full nucleotide / singleton width, so the model is
                # equivalent to a Conv1d over the sequence axis. Squeeze the width dimension
                # and permute to torch (out_channels, in_channels, kernel).
                if kernel.shape[1] == 1:
                    # conv2: width is the singleton dimension.
                    kernel = kernel[:, 0, :, :].permute(2, 1, 0).contiguous()
                else:
                    # conv1: the second axis is the 4-nucleotide channel, in_channels == 1.
                    kernel = kernel[:, :, 0, :].permute(2, 1, 0).contiguous()
            else:
                # Keras Dense kernel: (in, out) -> torch Linear (out, in).
                kernel = kernel.t().contiguous()
            state_dict[f"{torch_prefix}.weight"] = kernel
            state_dict[f"{torch_prefix}.bias"] = bias
    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = os.path.join(os.path.dirname(__file__), "saved_models")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
