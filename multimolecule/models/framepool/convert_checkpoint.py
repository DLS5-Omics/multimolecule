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

# The upstream Framepool weights are distributed by Karollus et al. (2021,
# PLOS Comput. Biol., 10.1371/journal.pcbi.1008982) via Zenodo:
# https://zenodo.org/record/3584238/files/Framepool_combined_residual.h5
# `h5py` is a *conversion-only* dependency; the converted MultiMolecule model
# is pure-torch and never imports it at runtime.


from __future__ import annotations

import os
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import FramepoolConfig as Config
from multimolecule.models import FramepoolForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Framepool one-hot order is ["A", "C", "G", "T"] (UTRVariantEffectModel nuc_dict).
# MultiMolecule exposes the 5'UTR alphabet as RNA, so the upstream T channel is mapped to U.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "U"]

# Released checkpoint filename on Zenodo record 3584238.
DEFAULT_CHECKPOINT = "Framepool_combined_residual.h5"

# Map upstream Keras layer names to MultiMolecule parameter prefixes.
# Encoder convolutions: ``convolution_<i>`` ↔ ``encoder.layers.<i>.conv``.
# Prediction head: ``fully_connected_<i>`` ↔ ``prediction.dense.<i>.dense``;
# ``mrl_output_unscaled`` ↔ ``prediction.unscaled``; ``scaling_regression`` ↔ ``prediction.scaling``.
ENCODER_LAYERS = ("convolution_0", "convolution_1", "convolution_2")
DENSE_LAYERS = ("fully_connected_0",)
UNSCALED_LAYER = "mrl_output_unscaled"
SCALING_LAYER = "scaling_regression"


def convert_checkpoint(convert_config):
    print(f"Converting Framepool checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    ckpt = convert_config.checkpoint_path
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, DEFAULT_CHECKPOINT)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = None
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(ckpt)
    # The first encoder convolution sees the one-hot input; permute its input channels from upstream
    # ``ACGT`` order to the MultiMolecule RNA tokenizer order, with upstream T exposed as U.
    key = "encoder.layers.0.conv.weight"
    weight = state_dict[key]
    state_dict[key] = convert_one_hot_embeddings(
        weight,
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    state_dict = OrderedDict(
        (key if key.startswith("prediction.") else f"model.{key}", value) for key, value in state_dict.items()
    )
    expected = model.state_dict()
    for key in expected:
        if key not in state_dict and not key.startswith(("model.embeddings.",)):
            raise KeyError(f"Converted state dict is missing the learned parameter {key!r}.")

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(file: str) -> OrderedDict:
    """Read the legacy Keras (.h5) Framepool weights directly via h5py.

    Keras 2.x stores Conv1D kernels as ``(kernel, in_channels, out_channels)`` and Dense kernels as
    ``(in, out)``; both are converted to the corresponding torch layouts here.
    """
    if not file or not os.path.exists(file):
        raise FileNotFoundError(
            "No upstream Framepool checkpoint found. Download "
            "https://zenodo.org/record/3584238/files/Framepool_combined_residual.h5 and pass it (or its containing "
            "directory) via `--checkpoint_path`."
        )
    import h5py  # noqa: PLC0415  conversion-only dependency

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as f:
        weights = f["model_weights"]
        for index, layer_name in enumerate(ENCODER_LAYERS):
            kernel, bias = _read_layer(weights, layer_name)
            # Keras Conv1D kernel: (kernel, in_channels, out_channels) → torch (out, in, kernel).
            kernel = kernel.permute(2, 1, 0).contiguous()
            state_dict[f"encoder.layers.{index}.conv.weight"] = kernel
            state_dict[f"encoder.layers.{index}.conv.bias"] = bias
        for index, layer_name in enumerate(DENSE_LAYERS):
            kernel, bias = _read_layer(weights, layer_name)
            # Keras Dense kernel: (in, out) → torch Linear (out, in).
            state_dict[f"prediction.dense.{index}.dense.weight"] = kernel.t().contiguous()
            state_dict[f"prediction.dense.{index}.dense.bias"] = bias
        kernel, bias = _read_layer(weights, UNSCALED_LAYER)
        state_dict["prediction.unscaled.weight"] = kernel.t().contiguous()
        state_dict["prediction.unscaled.bias"] = bias
        kernel, _ = _read_layer(weights, SCALING_LAYER, has_bias=False)
        state_dict["prediction.scaling.weight"] = kernel.t().contiguous()
    return state_dict


def _read_layer(weights, name: str, has_bias: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Keras stores per-layer weights at ``weights[name][<weight_path>]``. The intermediate group is named
    # after the layer with a ``_<n>`` suffix appended by Keras on graph rebuilds (e.g. ``convolution_0_2``),
    # so we resolve the actual path through the ``weight_names`` attribute rather than hard-coding it.
    group = weights[name]
    weight_names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs["weight_names"]]
    tensors: dict[str, torch.Tensor] = {}
    for weight_name in weight_names:
        dataset = group
        for part in weight_name.split("/"):
            dataset = dataset[part]
        kind = weight_name.split("/")[-1].split(":")[0]
        tensors[kind] = torch.from_numpy(dataset[()]).float()
    kernel = tensors["kernel"]
    bias = tensors.get("bias") if has_bias else None
    return kernel, bias


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = os.path.join(os.path.dirname(__file__), "saved_models")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
