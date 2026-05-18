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
import numpy as np
import torch

from multimolecule.models import BassetConfig as Config
from multimolecule.models import BassetForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Basset (davek44/Basset, Kelley, Snoek & Rinn, Genome Research 2016) one-hot
# encodes DNA as ["A", "C", "G", "T"]. The released pretrained model is the Lua-Torch checkpoint
# `pretrained_model.th` (`gunzip` of `pretrained_model.th.gz` shipped via `install_data.py`). The
# `.th` file is read with the conversion-only `torchfile` dependency.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# The upstream module is a flat ``nn.Sequential``:
#   [SpatialConvolution, SpatialBatchNormalization, ReLU, SpatialMaxPooling] x3
#   -> Reshape
#   -> [Linear, BatchNormalization, ReLU, Dropout] x2
#   -> Linear -> Sigmoid
# Each parameterised module maps 1:1 onto the MultiMolecule module tree below.
# The target model is ``BassetForSequencePrediction``; its backbone lives under ``model.`` and
# the 164-way DNase-I classifier is the ``sequence_head.decoder`` Linear.
CONV_LAYER_KEYS = [
    "model.encoder.layers.0",
    "model.encoder.layers.1",
    "model.encoder.layers.2",
]
FC_LAYER_KEYS = [
    "model.encoder.fc_layers.0",
    "model.encoder.fc_layers.1",
]
DECODER_KEY = "sequence_head.decoder"


def convert_checkpoint(convert_config):
    print(f"Converting Basset checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    modules = _load_original_modules(convert_config.checkpoint_path)
    state_dict = _convert_checkpoint(modules, config)
    key = f"{CONV_LAYER_KEYS[0]}.conv.weight"
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


def _load_original_modules(checkpoint_path: str) -> list:
    """Load the upstream Lua-Torch ``.th`` and return its flat list of ``nn.Sequential`` modules.

    The original Basset model is distributed by davek44/Basset as a Torch7 ``pretrained_model.th``
    (download + ``gunzip`` via ``install_data.py``; weights are not redistributed here). Pass the
    decompressed ``.th`` via ``--checkpoint_path``.
    """
    if not (checkpoint_path and os.path.isfile(checkpoint_path)):
        raise FileNotFoundError(
            "Upstream Basset checkpoint not found. Download davek44/Basset's "
            "`pretrained_model.th.gz`, `gunzip` it, and pass the resulting `.th` "
            "via `--checkpoint_path`."
        )
    try:
        import torchfile
    except ImportError as error:
        raise ImportError(
            "Reading the Lua-Torch Basset checkpoint requires the conversion-only dependency "
            "`torchfile` (`pip install torchfile`)."
        ) from error

    obj = torchfile.load(checkpoint_path)
    model = obj[b"model"] if isinstance(obj, dict) and b"model" in obj else obj
    modules = model[b"modules"]
    return list(modules)


def _to_tensor(array) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)


def _torch_typename(module) -> str:
    name = module.torch_typename()
    return name.decode() if isinstance(name, bytes) else str(name)


def _module_attr(module, key: bytes):
    return module._obj[key]


def _convert_conv(module) -> dict:
    # Torch7 SpatialConvolution weight: (out, in, kH=1, kW) -> PyTorch Conv1d: (out, in, kW).
    weight = _to_tensor(_module_attr(module, b"weight"))
    if weight.dim() == 4:
        weight = weight.squeeze(2)
    bias = _to_tensor(_module_attr(module, b"bias"))
    return {"conv.weight": weight, "conv.bias": bias}


def _convert_batch_norm(module, eps: float) -> dict:
    # Torch7 (Spatial)BatchNormalization stores `running_std = 1 / sqrt(running_var + eps)`,
    # whereas PyTorch BatchNorm stores `running_var` and recomputes `1 / sqrt(var + eps)`.
    running_mean = _to_tensor(_module_attr(module, b"running_mean"))
    running_std = _to_tensor(_module_attr(module, b"running_std"))
    running_var = running_std.pow(-2.0) - eps
    return {
        "batch_norm.weight": _to_tensor(_module_attr(module, b"weight")),
        "batch_norm.bias": _to_tensor(_module_attr(module, b"bias")),
        "batch_norm.running_mean": running_mean,
        "batch_norm.running_var": running_var,
    }


def _convert_linear(module) -> dict:
    return {
        "weight": _to_tensor(_module_attr(module, b"weight")),
        "bias": _to_tensor(_module_attr(module, b"bias")),
    }


def _convert_checkpoint(modules: list, config: Config) -> OrderedDict:
    eps = config.batch_norm_eps
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    conv_blocks: list[tuple] = []
    fc_blocks: list[tuple] = []
    decoder = None

    pending_conv = None
    pending_linear = None
    for module in modules:
        typename = _torch_typename(module)
        if typename.endswith("SpatialConvolution"):
            pending_conv = module
        elif typename.endswith("SpatialBatchNormalization"):
            conv_blocks.append((pending_conv, module))
            pending_conv = None
        elif typename.endswith("Linear"):
            pending_linear = module
        elif typename.endswith("BatchNormalization"):  # plain (non-spatial) BN follows an FC Linear
            fc_blocks.append((pending_linear, module))
            pending_linear = None

    if pending_linear is not None:  # final classifier Linear has no trailing BatchNorm
        decoder = pending_linear

    if len(conv_blocks) != len(CONV_LAYER_KEYS):
        raise ValueError(f"Expected {len(CONV_LAYER_KEYS)} conv blocks, found {len(conv_blocks)}.")
    if len(fc_blocks) != len(FC_LAYER_KEYS):
        raise ValueError(f"Expected {len(FC_LAYER_KEYS)} fully-connected blocks, found {len(fc_blocks)}.")
    if decoder is None:
        raise ValueError("Could not locate the final Basset classifier Linear layer.")

    for prefix, (conv, bn) in zip(CONV_LAYER_KEYS, conv_blocks):
        for suffix, value in _convert_conv(conv).items():
            state_dict[f"{prefix}.{suffix}"] = value
        for suffix, value in _convert_batch_norm(bn, eps).items():
            state_dict[f"{prefix}.{suffix}"] = value

    for prefix, (linear, bn) in zip(FC_LAYER_KEYS, fc_blocks):
        for suffix, value in _convert_linear(linear).items():
            state_dict[f"{prefix}.dense.{suffix}"] = value
        for suffix, value in _convert_batch_norm(bn, eps).items():
            state_dict[f"{prefix}.{suffix}"] = value

    for suffix, value in _convert_linear(decoder).items():
        state_dict[f"{DECODER_KEY}.{suffix}"] = value

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
