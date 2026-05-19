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

from multimolecule.models import ScBassetConfig as Config
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.models.scbasset.modeling_scbasset import (
    ScBassetForSequencePrediction,
)
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream scBasset (Yuan & Kelley 2022, calico/scBasset) one-hot encodes DNA as ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# The shipped checkpoint is the Buenrostro2018 hematopoiesis tutorial model (`buen_model_sc.h5`,
# https://storage.googleapis.com/scbasset_tutorial_data/buen_model_sc.h5), which has 2034 single cells.
# scBasset's final cell-embedding layer is dataset-specific (one row per cell), so there is no
# dataset-independent foundation checkpoint; this single instance is documented in the README.


def convert_checkpoint(convert_config):
    print(f"Converting scBasset checkpoint at {convert_config.checkpoint_path}")
    config = Config(num_labels=_infer_num_labels(convert_config.checkpoint_path))
    model = ScBassetForSequencePrediction(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, config)
    key = "model.encoder.layers.0.conv.weight"
    weight = state_dict.get(key)
    if weight is not None:
        state_dict[key] = convert_one_hot_embeddings(
            weight,
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


def _read_weight(f: h5py.File, layer: str, name: str) -> torch.Tensor:
    # Legacy Keras 2.6 (.h5) cannot be deserialized by modern Keras, so the raw HDF5 datasets are read directly.
    # `model.save_weights` stores each layer's tensors under /<layer>/<layer>/<name>:0.
    key = f"{layer}/{layer}/{name}:0"
    if key not in f:
        raise KeyError(f"Expected weight {key} not found in scBasset checkpoint; refusing to skip conversion.")
    return torch.from_numpy(f[key][()])


def _infer_num_labels(checkpoint_path: str) -> int:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Upstream scBasset checkpoint not found. Download the canonical Buenrostro2018 "
            "`buen_model_sc.h5` checkpoint and pass it via `--checkpoint_path`."
        )
    with h5py.File(checkpoint_path, "r") as f:
        return int(_read_weight(f, "dense_1", "bias").numel())


def _convert_checkpoint(checkpoint_path: str, config: Config) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(checkpoint_path, "r") as f:
        # Stem convolution: Keras Conv1D kernel (kw, in, out) -> torch Conv1d (out, in, kw); use_bias=False.
        state_dict["model.encoder.layers.0.conv.weight"] = _read_weight(f, "conv1d", "kernel").permute(2, 1, 0)
        _convert_batch_norm(f, "batch_normalization", "model.encoder.layers.0.batch_norm", state_dict)

        # Reducing convolution tower: conv1d_1 .. conv1d_N -> encoder.layers.1 .. encoder.layers.N.
        for idx, _ in enumerate(config.tower_channels):
            layer = f"conv1d_{idx + 1}"
            state_dict[f"model.encoder.layers.{idx + 1}.conv.weight"] = _read_weight(f, layer, "kernel").permute(
                2, 1, 0
            )
            _convert_batch_norm(
                f,
                f"batch_normalization_{idx + 1}",
                f"model.encoder.layers.{idx + 1}.batch_norm",
                state_dict,
            )

        # Final pointwise convolution (kernel size 1).
        last = len(config.tower_channels) + 1
        state_dict[f"model.encoder.layers.{last}.conv.weight"] = _read_weight(f, f"conv1d_{last}", "kernel").permute(
            2, 1, 0
        )
        _convert_batch_norm(
            f,
            f"batch_normalization_{last}",
            f"model.encoder.layers.{last}.batch_norm",
            state_dict,
        )

        # Dense bottleneck: Keras Dense kernel (in, out) -> torch Linear (out, in); use_bias=False (batch_norm).
        bottleneck_bn = f"batch_normalization_{last + 1}"
        state_dict["model.encoder.bottleneck.dense.weight"] = _read_weight(f, "dense", "kernel").t()
        _convert_batch_norm(f, bottleneck_bn, "model.encoder.bottleneck.batch_norm", state_dict)

        # Cell-embedding (final dense) layer -> shared SequencePredictionHead decoder. This layer is
        # DATASET-SPECIFIC: one row per single cell in the training atlas.
        state_dict["sequence_head.decoder.weight"] = _read_weight(f, "dense_1", "kernel").t()
        state_dict["sequence_head.decoder.bias"] = _read_weight(f, "dense_1", "bias")

    return state_dict


def _convert_batch_norm(f: h5py.File, layer: str, prefix: str, state_dict: OrderedDict) -> None:
    state_dict[f"{prefix}.weight"] = _read_weight(f, layer, "gamma")
    state_dict[f"{prefix}.bias"] = _read_weight(f, layer, "beta")
    state_dict[f"{prefix}.running_mean"] = _read_weight(f, layer, "moving_mean")
    state_dict[f"{prefix}.running_var"] = _read_weight(f, layer, "moving_variance")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
