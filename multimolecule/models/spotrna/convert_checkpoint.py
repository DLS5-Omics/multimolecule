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

import argparse
import os

import numpy as np
import torch

from multimolecule.models import SpotRnaConfig as Config
from multimolecule.models import SpotRnaModel
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_

_CHANNEL_PERM = [0, 2, 3, 1, 4, 6, 7, 5]


def convert_checkpoint(checkpoint_dir: str, output_dir: str) -> None:
    import tensorflow as tf

    from multimolecule.tokenisers import RnaTokenizer
    from multimolecule.tokenisers.rna.utils import get_alphabet, get_tokenizer_config

    config = Config()
    model = SpotRnaModel(config)

    state_dict = {}
    state_dict["input_mean"] = model.input_mean[:, :, :, _CHANNEL_PERM]
    state_dict["input_std"] = model.input_std[:, :, :, _CHANNEL_PERM]
    base_mean = state_dict["input_mean"]
    base_std = state_dict["input_std"]

    for net_idx, net_config in enumerate(config.networks):
        checkpoint_path = os.path.join(checkpoint_dir, f"model{net_idx}")
        reader = tf.train.load_checkpoint(checkpoint_path)
        net_state = _convert_network_state_dict(net_config, reader)
        net_mean, net_std = _load_input_stats(tf, checkpoint_path)
        net_mean = net_mean[:, :, :, _CHANNEL_PERM]
        net_std = net_std[:, :, :, _CHANNEL_PERM]
        # Keep the shared preprocessing and fold checkpoint-specific input stats into an affine correction.
        net_state["input_scale"] = base_std / net_std
        net_state["input_bias"] = (base_mean - net_mean) / net_std
        for key, value in net_state.items():
            state_dict[f"networks.{net_idx}.{key}"] = value

    state_dict = {k: v.float() for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    tokenizer_config = get_tokenizer_config()
    tokenizer_config["alphabet"] = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    tokenizer = RnaTokenizer(**tokenizer_config)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def _convert_network_state_dict(net_config, reader):
    state_dict = {}

    init_weight = _convert_conv2d_kernel(reader.get_tensor("initconv/conv2d/kernel"))
    state_dict["initial_conv.weight"] = init_weight[:, _CHANNEL_PERM, :, :]
    state_dict["initial_conv.bias"] = torch.from_numpy(reader.get_tensor("initconv/conv2d/bias"))

    for block_idx in range(net_config.num_conv_blocks):
        conv3_idx = block_idx * 2
        conv5_idx = block_idx * 2 + 1
        prefix = f"conv_blocks.{block_idx}"

        state_dict[f"{prefix}.conv1.weight"] = _convert_conv2d_kernel(
            reader.get_tensor(f"conv{conv3_idx}/conv2d/kernel")
        )
        state_dict[f"{prefix}.conv1.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv3_idx}/conv2d/bias"))
        state_dict[f"{prefix}.norm1.weight"] = torch.from_numpy(reader.get_tensor(f"conv{conv3_idx}/LayerNorm/gamma"))
        state_dict[f"{prefix}.norm1.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv3_idx}/LayerNorm/beta"))

        state_dict[f"{prefix}.conv2.weight"] = _convert_conv2d_kernel(
            reader.get_tensor(f"conv{conv5_idx}/conv2d/kernel")
        )
        state_dict[f"{prefix}.conv2.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv5_idx}/conv2d/bias"))
        state_dict[f"{prefix}.norm2.weight"] = torch.from_numpy(reader.get_tensor(f"conv{conv5_idx}/LayerNorm/gamma"))
        state_dict[f"{prefix}.norm2.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv5_idx}/LayerNorm/beta"))

    state_dict["output_norm.weight"] = torch.from_numpy(reader.get_tensor("outputactnorm/LayerNorm/gamma"))
    state_dict["output_norm.bias"] = torch.from_numpy(reader.get_tensor("outputactnorm/LayerNorm/beta"))

    if net_config.num_blstm_blocks > 0:
        h = net_config.blstm_hidden_size

        fw = _convert_lstm_weights(
            reader.get_tensor("brnn1/bidirectional_rnn/fw/basic_lstm_cell/kernel"),
            reader.get_tensor("brnn1/bidirectional_rnn/fw/basic_lstm_cell/bias"),
            h,
        )
        state_dict["blstm.weight_ih_l0"] = fw["weight_ih"]
        state_dict["blstm.weight_hh_l0"] = fw["weight_hh"]
        state_dict["blstm.bias_ih_l0"] = fw["bias_ih"]
        state_dict["blstm.bias_hh_l0"] = fw["bias_hh"]

        bw = _convert_lstm_weights(
            reader.get_tensor("brnn1/bidirectional_rnn/bw/basic_lstm_cell/kernel"),
            reader.get_tensor("brnn1/bidirectional_rnn/bw/basic_lstm_cell/bias"),
            h,
        )
        state_dict["blstm.weight_ih_l0_reverse"] = bw["weight_ih"]
        state_dict["blstm.weight_hh_l0_reverse"] = bw["weight_hh"]
        state_dict["blstm.bias_ih_l0_reverse"] = bw["bias_ih"]
        state_dict["blstm.bias_hh_l0_reverse"] = bw["bias_hh"]

    for fc_idx in range(net_config.num_fc_blocks):
        tf_prefix = f"Hidden_FC_{fc_idx}"
        pt_prefix = f"fc_blocks.{fc_idx}"

        state_dict[f"{pt_prefix}.fc.weight"] = _convert_fc_weights(
            reader.get_tensor(f"{tf_prefix}/fully_connected/weights")
        )
        state_dict[f"{pt_prefix}.fc.bias"] = torch.from_numpy(reader.get_tensor(f"{tf_prefix}/fully_connected/biases"))
        state_dict[f"{pt_prefix}.norm.weight"] = torch.from_numpy(reader.get_tensor(f"{tf_prefix}/LayerNorm/gamma"))
        state_dict[f"{pt_prefix}.norm.bias"] = torch.from_numpy(reader.get_tensor(f"{tf_prefix}/LayerNorm/beta"))

    state_dict["output_fc.weight"] = _convert_fc_weights(reader.get_tensor("output_FC/fully_connected/weights"))
    state_dict["output_fc.bias"] = torch.from_numpy(reader.get_tensor("output_FC/fully_connected/biases"))

    return state_dict


def _load_input_stats(tf, checkpoint_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.train.import_meta_graph(checkpoint_path + ".meta", clear_devices=True)
        graph = tf.compat.v1.get_default_graph()
        input_mean = torch.from_numpy(sess.run(graph.get_tensor_by_name("Const:0")))
        input_std = torch.from_numpy(sess.run(graph.get_tensor_by_name("Const_1:0")))
    return input_mean, input_std


def _convert_conv2d_kernel(tf_kernel: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(tf_kernel.transpose(3, 2, 0, 1).copy())


def _convert_fc_weights(tf_weights: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(tf_weights.T.copy())


def _convert_lstm_weights(tf_kernel: np.ndarray, tf_bias: np.ndarray, hidden_size: int) -> dict[str, torch.Tensor]:
    input_size = tf_kernel.shape[0] - hidden_size
    w_ih = tf_kernel[:input_size, :]
    w_hh = tf_kernel[input_size:, :]

    def _reorder_gates(w: np.ndarray) -> np.ndarray:
        i, j, f, o = np.split(w, 4, axis=-1)
        return np.concatenate([i, f, j, o], axis=-1)

    # TF BasicLSTMCell adds forget_bias at runtime, so fold it into the stored bias here.
    bias_with_forget = tf_bias.copy()
    h = hidden_size
    bias_with_forget[2 * h : 3 * h] += 1.0

    return {
        "weight_ih": torch.from_numpy(_reorder_gates(w_ih).T.copy()),
        "weight_hh": torch.from_numpy(_reorder_gates(w_hh).T.copy()),
        "bias_ih": torch.from_numpy(_reorder_gates(bias_with_forget).copy()),
        "bias_hh": torch.zeros(4 * hidden_size),
    }


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
