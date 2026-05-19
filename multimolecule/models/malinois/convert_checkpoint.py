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

import io
import os
import tarfile
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import MalinoisConfig as Config
from multimolecule.models import MalinoisForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Malinois (sjgosai/boda2) one-hot encodes DNA with `boda.common.constants.STANDARD_NT`,
# which is the order ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    print(f"Converting Malinois checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.input_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    checkpoint_path = convert_config.checkpoint_path
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        original_state_dict = _load_original_state_dict(checkpoint_path)
        state_dict = _convert_checkpoint(original_state_dict, config)
        key = "model.encoder.blocks.0.conv.weight"
        weight = state_dict.get(key)
        if weight is None:
            raise KeyError(f"Expected `{key}` in the converted Malinois state dict for channel conversion.")
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
            "No upstream Malinois checkpoint found. Download "
            "`malinois_artifacts__20211113_021200__287348.tar.gz` from "
            "`gs://tewhey-public-data/CODA_resources/` (publicly mirrored at "
            "`https://storage.googleapis.com/tewhey-public-data/CODA_resources/`) and pass the tarball, "
            "the unpacked directory, or `torch_checkpoint.pt` via `--checkpoint_path`."
        )

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _load_original_state_dict(checkpoint_path: str) -> OrderedDict:
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "torch_checkpoint.pt")
    if tarfile.is_tarfile(checkpoint_path):
        with tarfile.open(checkpoint_path) as tar:
            torch_checkpoint = next(
                (
                    member
                    for member in tar.getmembers()
                    if member.isfile() and os.path.basename(member.name) == "torch_checkpoint.pt"
                ),
                None,
            )
            if torch_checkpoint is None:
                raise FileNotFoundError("Could not find `torch_checkpoint.pt` inside the Malinois artifact tarball.")
            file = tar.extractfile(torch_checkpoint)
            if file is None:
                raise FileNotFoundError("Could not read `torch_checkpoint.pt` from the Malinois artifact tarball.")
            checkpoint = torch.load(io.BytesIO(file.read()), map_location="cpu", weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return OrderedDict(checkpoint["model_state_dict"])
    return OrderedDict(checkpoint)


def _convert_checkpoint(original_state_dict: OrderedDict, config: Config) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    # Convolutional encoder: conv{i}.conv.* -> encoder.blocks.{i-1}.conv.*,
    # conv{i}.bn_layer.* -> encoder.blocks.{i-1}.norm.*.
    for index in range(config.num_conv_layers):
        src = f"conv{index + 1}"
        dst = f"model.encoder.blocks.{index}"
        _require(original_state_dict, f"{src}.conv.weight")
        state_dict[f"{dst}.conv.weight"] = original_state_dict[f"{src}.conv.weight"]
        state_dict[f"{dst}.conv.bias"] = original_state_dict[f"{src}.conv.bias"]
        _convert_batchnorm(original_state_dict, f"{src}.bn_layer", state_dict, f"{dst}.norm")

    # Shared fully-connected layers: linear{i}.linear.* -> pooler.layers.{i-1}.dense.*,
    # linear{i}.bn_layer.* -> pooler.layers.{i-1}.norm.*.
    for index in range(config.num_linear_layers):
        src = f"linear{index + 1}"
        dst = f"model.pooler.layers.{index}"
        _require(original_state_dict, f"{src}.linear.weight")
        state_dict[f"{dst}.dense.weight"] = original_state_dict[f"{src}.linear.weight"]
        state_dict[f"{dst}.dense.bias"] = original_state_dict[f"{src}.linear.bias"]
        _convert_batchnorm(original_state_dict, f"{src}.bn_layer", state_dict, f"{dst}.norm")

    # Branched tower: branched.branched_layer_{i}.* -> pooler.branched.layers.{i-1}.linear.*.
    for index in range(config.num_branched_layers):
        src = f"branched.branched_layer_{index + 1}"
        dst = f"model.pooler.branched.layers.{index}.linear"
        _require(original_state_dict, f"{src}.weight")
        state_dict[f"{dst}.weight"] = original_state_dict[f"{src}.weight"]
        state_dict[f"{dst}.bias"] = original_state_dict[f"{src}.bias"]

    # The upstream `output` GroupedLinear (in_group_size=branched_channels, out_group_size=1,
    # groups=num_labels) is block-diagonal: output[g] only sees branch g's features. It maps the
    # branch-major (num_labels * branched_channels) representation to num_labels scalars. We fold
    # it into MultiMolecule's shared `sequence_head.decoder` as a dense block-diagonal
    # `Linear(num_labels * branched_channels, num_labels)`, which is numerically identical.
    _require(original_state_dict, "output.weight")
    output_weight = original_state_dict["output.weight"]  # (groups, branched_channels, 1)
    output_bias = original_state_dict["output.bias"]  # (groups, 1, 1)
    groups, branched_channels, _ = output_weight.shape
    decoder_weight = torch.zeros(groups, groups * branched_channels, dtype=output_weight.dtype)
    decoder_bias = torch.zeros(groups, dtype=output_bias.dtype)
    for g in range(groups):
        decoder_weight[g, g * branched_channels : (g + 1) * branched_channels] = output_weight[g, :, 0]
        decoder_bias[g] = output_bias[g, 0, 0]
    state_dict["sequence_head.decoder.weight"] = decoder_weight.contiguous()
    state_dict["sequence_head.decoder.bias"] = decoder_bias.contiguous()

    return state_dict


def _convert_batchnorm(original_state_dict: OrderedDict, src: str, state_dict: OrderedDict, dst: str) -> None:
    _require(original_state_dict, f"{src}.weight")
    state_dict[f"{dst}.weight"] = original_state_dict[f"{src}.weight"]
    state_dict[f"{dst}.bias"] = original_state_dict[f"{src}.bias"]
    state_dict[f"{dst}.running_mean"] = original_state_dict[f"{src}.running_mean"]
    state_dict[f"{dst}.running_var"] = original_state_dict[f"{src}.running_var"]


def _require(state_dict: OrderedDict, key: str) -> None:
    if key not in state_dict:
        raise KeyError(f"Expected key `{key}` in the upstream Malinois checkpoint but it was not found.")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
