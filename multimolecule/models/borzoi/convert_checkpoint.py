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
import re
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import BorzoiConfig as Config
from multimolecule.models import BorzoiForTokenPrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import (
    convert_one_hot_embeddings,
    load_checkpoint,
    save_checkpoint,
)
from multimolecule.tokenisers.dna.utils import (
    convert_word_embeddings,
    get_alphabet,
    get_tokenizer_config,
)

torch.manual_seed(1016)


# Upstream Borzoi one-hot encodes DNA in the order ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

NUM_TRANSFORMER_LAYERS = 8


def _layer_name(prefix: str, index: int) -> str:
    return prefix if index == 0 else f"{prefix}_{index}"


def _read_layer(weights, layer: str, variable: str) -> torch.Tensor:
    return torch.from_numpy(weights[layer][layer][f"{variable}:0"][()])


def _read_mha(weights, layer: str, *path: str) -> torch.Tensor:
    group = weights[layer]
    for part in path:
        group = group[part]
    return torch.from_numpy(group[()])


def _keras_conv1d_to_torch(weight: torch.Tensor) -> torch.Tensor:
    return weight.permute(2, 1, 0).contiguous()


def _keras_dense_to_torch(weight: torch.Tensor) -> torch.Tensor:
    return weight.t().contiguous()


def _keras_dense_to_conv1d(weight: torch.Tensor) -> torch.Tensor:
    return weight.t().unsqueeze(-1).contiguous()


def _keras_depthwise_to_torch(weight: torch.Tensor) -> torch.Tensor:
    return weight.squeeze(-1).t().unsqueeze(1).contiguous()


def _keras_pointwise_to_torch(weight: torch.Tensor) -> torch.Tensor:
    return weight.squeeze(0).t().unsqueeze(-1).contiguous()


def _add_batch_norm(state_dict: OrderedDict[str, torch.Tensor], weights, layer: str, prefix: str) -> None:
    state_dict[f"{prefix}.weight"] = _read_layer(weights, layer, "gamma")
    state_dict[f"{prefix}.bias"] = _read_layer(weights, layer, "beta")
    state_dict[f"{prefix}.running_mean"] = _read_layer(weights, layer, "moving_mean")
    state_dict[f"{prefix}.running_var"] = _read_layer(weights, layer, "moving_variance")


def _add_layer_norm(state_dict: OrderedDict[str, torch.Tensor], weights, layer: str, prefix: str) -> None:
    state_dict[f"{prefix}.weight"] = _read_layer(weights, layer, "gamma")
    state_dict[f"{prefix}.bias"] = _read_layer(weights, layer, "beta")


def _add_conv1d(state_dict: OrderedDict[str, torch.Tensor], weights, layer: str, prefix: str) -> None:
    state_dict[f"{prefix}.weight"] = _keras_conv1d_to_torch(_read_layer(weights, layer, "kernel"))
    state_dict[f"{prefix}.bias"] = _read_layer(weights, layer, "bias")


def _add_dense(state_dict: OrderedDict[str, torch.Tensor], weights, layer: str, prefix: str) -> None:
    state_dict[f"{prefix}.weight"] = _keras_dense_to_torch(_read_layer(weights, layer, "kernel"))
    state_dict[f"{prefix}.bias"] = _read_layer(weights, layer, "bias")


def _add_dense_conv1d(state_dict: OrderedDict[str, torch.Tensor], weights, layer: str, prefix: str) -> None:
    state_dict[f"{prefix}.weight"] = _keras_dense_to_conv1d(_read_layer(weights, layer, "kernel"))
    state_dict[f"{prefix}.bias"] = _read_layer(weights, layer, "bias")


def _add_separable_conv1d(state_dict: OrderedDict[str, torch.Tensor], weights, layer: str, prefix: str) -> None:
    state_dict[f"{prefix}.depthwise.weight"] = _keras_depthwise_to_torch(
        _read_layer(weights, layer, "depthwise_kernel")
    )
    state_dict[f"{prefix}.pointwise.weight"] = _keras_pointwise_to_torch(
        _read_layer(weights, layer, "pointwise_kernel")
    )
    state_dict[f"{prefix}.pointwise.bias"] = _read_layer(weights, layer, "bias")


def _add_attention(state_dict: OrderedDict[str, torch.Tensor], weights, index: int) -> None:
    layer = _layer_name("multihead_attention", index)
    prefix = f"model.encoder.layers.{index}.attention"
    state_dict[f"{prefix}.rel_content_bias"] = _read_mha(weights, layer, "r_w_bias:0")
    state_dict[f"{prefix}.rel_pos_bias"] = _read_mha(weights, layer, "r_r_bias:0")
    state_dict[f"{prefix}.to_q.weight"] = _keras_dense_to_torch(_read_mha(weights, layer, layer, "q_layer", "kernel:0"))
    state_dict[f"{prefix}.to_k.weight"] = _keras_dense_to_torch(_read_mha(weights, layer, layer, "k_layer", "kernel:0"))
    state_dict[f"{prefix}.to_v.weight"] = _keras_dense_to_torch(_read_mha(weights, layer, layer, "v_layer", "kernel:0"))
    state_dict[f"{prefix}.to_rel_k.weight"] = _keras_dense_to_torch(
        _read_mha(weights, layer, layer, "r_k_layer", "kernel:0")
    )
    state_dict[f"{prefix}.to_out.weight"] = _keras_dense_to_torch(
        _read_mha(weights, layer, layer, "embedding_layer", "kernel:0")
    )
    state_dict[f"{prefix}.to_out.bias"] = _read_mha(weights, layer, layer, "embedding_layer", "bias:0")


def _head_layer(weights, num_labels: int) -> str:
    pattern = re.compile(r"^dense(?:_\d+)?$")
    for layer in weights:
        if pattern.match(layer) is None:
            continue
        if "bias:0" in weights[layer][layer] and weights[layer][layer]["bias:0"].shape == (num_labels,):
            return layer
    raise KeyError(f"Could not find Borzoi output head with {num_labels} labels in the checkpoint")


def _convert_checkpoint(file: str, species: str) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    if not file or not os.path.exists(file):
        raise FileNotFoundError(
            "No upstream Borzoi checkpoint found. Download the official Calico/Baskerville "
            "`model*_best.h5` checkpoint and pass it via `--checkpoint_path`."
        )

    import h5py  # noqa: PLC0415  # transient conversion-only dependency, never imported at runtime

    config = Config(species=species)

    with h5py.File(file, "r") as h5:
        weights = h5["model_weights"]

        conv_map = [
            ("conv1d", "model.encoder.stem.conv1"),
            ("conv1d_1", "model.encoder.conv_tower.0.conv1"),
            ("conv1d_2", "model.encoder.conv_tower.1.conv1"),
            ("conv1d_3", "model.encoder.conv_tower.2.conv1"),
            ("conv1d_4", "model.encoder.conv_tower.3.conv1"),
            ("conv1d_5", "model.encoder.conv_tower.4.conv1"),
            ("conv1d_6", "model.encoder.unet_bottleneck.conv1"),
            ("conv1d_7", "model.encoder.head.conv1"),
        ]
        for layer, prefix in conv_map:
            _add_conv1d(state_dict, weights, layer, prefix)

        batch_norm_map = [
            ("sync_batch_normalization", "model.encoder.conv_tower.0.batch_norm1"),
            ("sync_batch_normalization_1", "model.encoder.conv_tower.1.batch_norm1"),
            ("sync_batch_normalization_2", "model.encoder.conv_tower.2.batch_norm1"),
            ("sync_batch_normalization_3", "model.encoder.conv_tower.3.batch_norm1"),
            ("sync_batch_normalization_4", "model.encoder.conv_tower.4.batch_norm1"),
            ("sync_batch_normalization_5", "model.encoder.unet_bottleneck.batch_norm1"),
            ("sync_batch_normalization_6", "model.encoder.upsample1.batch_norm1"),
            ("sync_batch_normalization_7", "model.encoder.skip1_proj.batch_norm1"),
            ("sync_batch_normalization_8", "model.encoder.upsample0.batch_norm1"),
            ("sync_batch_normalization_9", "model.encoder.skip0_proj.batch_norm1"),
            ("sync_batch_normalization_10", "model.encoder.head.batch_norm1"),
        ]
        for layer, prefix in batch_norm_map:
            _add_batch_norm(state_dict, weights, layer, prefix)

        for index in range(NUM_TRANSFORMER_LAYERS):
            _add_layer_norm(
                state_dict,
                weights,
                _layer_name("layer_normalization", 2 * index),
                f"model.encoder.layers.{index}.attention.layer_norm",
            )
            _add_attention(state_dict, weights, index)
            _add_layer_norm(
                state_dict,
                weights,
                _layer_name("layer_normalization", 2 * index + 1),
                f"model.encoder.layers.{index}.intermediate.layer_norm",
            )
            _add_dense(
                state_dict,
                weights,
                _layer_name("dense", 2 * index),
                f"model.encoder.layers.{index}.intermediate.dense1",
            )
            _add_dense(
                state_dict,
                weights,
                _layer_name("dense", 2 * index + 1),
                f"model.encoder.layers.{index}.intermediate.dense2",
            )

        dense_conv_map = [
            ("dense_16", "model.encoder.upsample1.conv1"),
            ("dense_17", "model.encoder.skip1_proj.conv1"),
            ("dense_18", "model.encoder.upsample0.conv1"),
            ("dense_19", "model.encoder.skip0_proj.conv1"),
        ]
        for layer, prefix in dense_conv_map:
            _add_dense_conv1d(state_dict, weights, layer, prefix)

        _add_separable_conv1d(state_dict, weights, "separable_conv1d", "model.encoder.separable1")
        _add_separable_conv1d(state_dict, weights, "separable_conv1d_1", "model.encoder.separable0")

        _add_dense(
            state_dict,
            weights,
            _head_layer(weights, config.num_labels),
            "token_head.decoder",
        )

    return state_dict


def convert_checkpoint(convert_config):
    print(f"Converting Borzoi checkpoint at {convert_config.checkpoint_path}")
    config = Config(species=convert_config.species)
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    tokenizer_config["model_max_length"] = config.sequence_length

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, convert_config.species)
    key = "model.encoder.stem.conv1.weight"
    state_dict[key] = convert_one_hot_embeddings(
        state_dict[key],
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    reference_state = model.state_dict()
    for key, value in reference_state.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    species: str = "human"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
