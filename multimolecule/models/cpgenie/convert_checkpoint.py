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

from multimolecule.models import CpGenieConfig as Config
from multimolecule.models import CpGenieForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream CpGenie (Zeng & Gifford 2017, gifford-lab/CpGenie) one-hot encodes DNA as ["A", "C", "G", "T"]
# with Theano channels-first input shape `(4, 1, 1001)`.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# The upstream `seq_128x3_5_5_2f_simple` template is a flat Keras `Sequential`:
#   [Convolution2D(C, 1, 5, border_mode='same', activation='relu'),
#    MaxPooling2D(pool_size=(1, 5), strides=(1, 3))] x 3   with C in [128, 256, 512]
#   -> Flatten
#   -> Dense(64, activation='relu') -> Dropout
#   -> Dense(64, activation='relu') -> Dropout
#   -> Dense(2) -> Activation('softmax')
# Each parameterised layer maps 1:1 onto the MultiMolecule module tree below.
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
    print(f"Converting CpGenie checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path)
    key = f"{CONV_LAYER_KEYS[0]}.conv.weight"
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


def _convert_checkpoint(checkpoint_path: str) -> OrderedDict:
    """Convert an upstream CpGenie Keras `.h5` weight file to a MultiMolecule state dict.

    CpGenie distributes 50 per-cell-line models as a `CpGenie_models.tar.gz` bundle
    (http://gerv.csail.mit.edu/CpGenie_models.tar.gz). Each model is a Keras 1.x architecture JSON
    plus an H5 weight file (`*_bestmodel_weights.h5`). Pass the **weights** H5 via
    `--checkpoint_path`; the architecture is fully reconstructed from `CpGenieConfig`.
    """
    if not (checkpoint_path and os.path.isfile(checkpoint_path)):
        raise FileNotFoundError(
            "Upstream CpGenie checkpoint not found. Download `CpGenie_models.tar.gz` from "
            "http://gerv.csail.mit.edu/CpGenie_models.tar.gz, extract one cell-line model, and pass "
            "the `*_bestmodel_weights.h5` file via `--checkpoint_path`."
        )

    try:
        import h5py  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "Reading CpGenie's Keras checkpoint requires the conversion-only dependency `h5py` " "(`pip install h5py`)."
        ) from error

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(checkpoint_path, "r") as h5file:
        conv_groups = _list_layer_groups(h5file, prefix="convolution2d_")
        if len(conv_groups) < len(CONV_LAYER_KEYS):
            raise ValueError(
                f"Expected {len(CONV_LAYER_KEYS)} convolution groups in CpGenie checkpoint, "
                f"found {len(conv_groups)}: {conv_groups}."
            )
        dense_groups = _list_layer_groups(h5file, prefix="dense_")
        if len(dense_groups) < len(FC_LAYER_KEYS) + 1:
            raise ValueError(
                f"Expected {len(FC_LAYER_KEYS) + 1} dense groups in CpGenie checkpoint, "
                f"found {len(dense_groups)}: {dense_groups}."
            )

        for prefix, group in zip(CONV_LAYER_KEYS, conv_groups[: len(CONV_LAYER_KEYS)]):
            weights = _read_layer_weights(h5file, group)
            kernel, bias = _extract_kernel_bias(weights, group)
            # Keras (Theano channels-first) Convolution2D weight: (out, in, kH=1, kW) -> torch Conv1d
            # weight (out, in, kW). The H=1 axis is collapsed.
            if kernel.dim() == 4:
                if kernel.shape[2] != 1:
                    raise ValueError(
                        f"Unexpected CpGenie Convolution2D kernel for layer {group}: shape {tuple(kernel.shape)}; "
                        "the height dimension must be 1."
                    )
                kernel = kernel.squeeze(2)
            elif kernel.dim() != 3:
                raise ValueError(
                    f"Unexpected CpGenie Convolution2D kernel rank for layer {group}: shape {tuple(kernel.shape)}."
                )
            state_dict[f"{prefix}.conv.weight"] = kernel.contiguous()
            state_dict[f"{prefix}.conv.bias"] = bias

        for prefix, group in zip(FC_LAYER_KEYS, dense_groups[: len(FC_LAYER_KEYS)]):
            weights = _read_layer_weights(h5file, group)
            kernel, bias = _extract_kernel_bias(weights, group)
            # Keras Dense kernel: (in, out) -> torch Linear weight (out, in).
            state_dict[f"{prefix}.dense.weight"] = kernel.t().contiguous()
            state_dict[f"{prefix}.dense.bias"] = bias

        decoder_group = dense_groups[len(FC_LAYER_KEYS)]
        weights = _read_layer_weights(h5file, decoder_group)
        kernel, bias = _extract_kernel_bias(weights, decoder_group)
        state_dict[f"{DECODER_KEY}.weight"] = kernel.t().contiguous()
        state_dict[f"{DECODER_KEY}.bias"] = bias

    return state_dict


def _list_layer_groups(h5file, prefix: str) -> list[str]:
    """Return the Keras layer names with a given prefix, sorted by their integer suffix.

    Keras 1.x stores weight groups as `<layer_name>/<weight_name>` (and sometimes under a nested
    `model_weights/` group). CpGenie's `Sequential` produces `convolution2d_1, convolution2d_2, ...`
    and `dense_1, dense_2, ...`; this helper sorts by the numeric suffix to keep the original order.
    """
    container = h5file["model_weights"] if "model_weights" in h5file else h5file

    def _suffix(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except (IndexError, ValueError):
            return -1

    matched = [name for name in container.keys() if name.startswith(prefix)]
    return sorted(matched, key=_suffix)


def _read_layer_weights(h5file, layer_name: str) -> dict:
    """Read every leaf dataset for a Keras layer group as `{short_name: tensor}`."""
    import h5py  # noqa: PLC0415

    container = h5file["model_weights"] if "model_weights" in h5file else h5file
    group = container[layer_name]
    weights: dict[str, torch.Tensor] = {}

    def _collect(node):
        for key in node.keys():
            item = node[key]
            if isinstance(item, h5py.Group):
                _collect(item)
            else:
                short = key.split(":")[0]
                # Strip the legacy `<layer>_W` / `<layer>_b` Keras 1 suffixes so callers can ask for
                # canonical `W`/`b` keys regardless of the Keras serialization version.
                if short.endswith("_W"):
                    short = "W"
                elif short.endswith("_b"):
                    short = "b"
                weights[short] = torch.as_tensor(item[()])

    _collect(group)
    return weights


def _extract_kernel_bias(weights: dict, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (kernel, bias) pair for a Keras layer, accepting both Keras-1 and Keras-2 naming."""
    if "W" in weights and "b" in weights:
        return weights["W"], weights["b"]
    if "kernel" in weights and "bias" in weights:
        return weights["kernel"], weights["bias"]
    raise KeyError(
        f"Could not find weight/bias tensors for CpGenie layer {layer_name}; available keys: {sorted(weights)}."
    )


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
