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

import numpy as np
import torch

from multimolecule.models import SpotRnaConfig as Config
from multimolecule.models import SpotRnaModel
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import get_alphabet, get_tokenizer_config


def convert_checkpoint(convert_config) -> None:
    import tensorflow as tf

    print(f"Converting SPOT-RNA checkpoint at {convert_config.checkpoint_path}")

    config = Config()
    model = SpotRnaModel(config)

    vocab_list = get_alphabet("nucleobase", prepend_tokens=[]).vocabulary
    channel_permutation = [original_vocab_list.index(token) for token in vocab_list]
    channel_permutation += [len(original_vocab_list) + original_vocab_list.index(token) for token in vocab_list]

    state_dict = {}
    base_mean = model.input_mean
    base_std = model.input_std

    for member_index, module_config in enumerate(config.module_configs):
        checkpoint_path = os.path.join(convert_config.checkpoint_path, f"model{member_index}")
        reader = tf.train.load_checkpoint(checkpoint_path)
        member_state = _convert_module_state_dict(module_config, reader, channel_permutation)
        member_mean, member_std = _load_input_stats(tf, checkpoint_path)
        member_mean = member_mean[:, :, :, channel_permutation]
        member_std = member_std[:, :, :, channel_permutation]
        # Keep the shared preprocessing and fold checkpoint-specific input stats into an affine correction.
        member_state["input_scale"] = base_std / member_std
        member_state["input_bias"] = (base_mean - member_mean) / member_std
        for key, value in member_state.items():
            state_dict[f"members.{member_index}.{key}"] = value

    # Normalise tensor dtype to fp32: upstream NumPy weights load as fp64, while the
    # MultiMolecule checkpoint is published as fp32 to match the rest of the model zoo.
    state_dict = {k: v.float() for k, v in state_dict.items()}

    tokenizer_config = get_tokenizer_config().copy()
    tokenizer_config["alphabet"] = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_module_state_dict(module_config, reader, channel_permutation):
    state_dict = {}

    projection_weight = _convert_conv2d_kernel(reader.get_tensor("initconv/conv2d/kernel"))
    state_dict["projection.weight"] = projection_weight[:, channel_permutation, :, :]
    state_dict["projection.bias"] = torch.from_numpy(reader.get_tensor("initconv/conv2d/bias"))

    for block_index in range(module_config.num_conv_blocks):
        conv3_index = block_index * 2
        conv5_index = block_index * 2 + 1
        prefix = f"layers.{block_index}"

        state_dict[f"{prefix}.conv1.weight"] = _convert_conv2d_kernel(
            reader.get_tensor(f"conv{conv3_index}/conv2d/kernel")
        )
        state_dict[f"{prefix}.conv1.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv3_index}/conv2d/bias"))
        state_dict[f"{prefix}.norm1.weight"] = torch.from_numpy(reader.get_tensor(f"conv{conv3_index}/LayerNorm/gamma"))
        state_dict[f"{prefix}.norm1.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv3_index}/LayerNorm/beta"))

        state_dict[f"{prefix}.conv2.weight"] = _convert_conv2d_kernel(
            reader.get_tensor(f"conv{conv5_index}/conv2d/kernel")
        )
        state_dict[f"{prefix}.conv2.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv5_index}/conv2d/bias"))
        state_dict[f"{prefix}.norm2.weight"] = torch.from_numpy(reader.get_tensor(f"conv{conv5_index}/LayerNorm/gamma"))
        state_dict[f"{prefix}.norm2.bias"] = torch.from_numpy(reader.get_tensor(f"conv{conv5_index}/LayerNorm/beta"))

    state_dict["layer_norm.weight"] = torch.from_numpy(reader.get_tensor("outputactnorm/LayerNorm/gamma"))
    state_dict["layer_norm.bias"] = torch.from_numpy(reader.get_tensor("outputactnorm/LayerNorm/beta"))

    if module_config.num_blstm_blocks > 0:
        hidden_size = module_config.blstm_hidden_size

        forward_weights = _convert_lstm_weights(
            reader.get_tensor("brnn1/bidirectional_rnn/fw/basic_lstm_cell/kernel"),
            reader.get_tensor("brnn1/bidirectional_rnn/fw/basic_lstm_cell/bias"),
            hidden_size,
        )
        state_dict["bilstm.weight_ih_l0"] = forward_weights["weight_ih"]
        state_dict["bilstm.weight_hh_l0"] = forward_weights["weight_hh"]
        state_dict["bilstm.bias_ih_l0"] = forward_weights["bias_ih"]
        state_dict["bilstm.bias_hh_l0"] = forward_weights["bias_hh"]

        backward_weights = _convert_lstm_weights(
            reader.get_tensor("brnn1/bidirectional_rnn/bw/basic_lstm_cell/kernel"),
            reader.get_tensor("brnn1/bidirectional_rnn/bw/basic_lstm_cell/bias"),
            hidden_size,
        )
        state_dict["bilstm.weight_ih_l0_reverse"] = backward_weights["weight_ih"]
        state_dict["bilstm.weight_hh_l0_reverse"] = backward_weights["weight_hh"]
        state_dict["bilstm.bias_ih_l0_reverse"] = backward_weights["bias_ih"]
        state_dict["bilstm.bias_hh_l0_reverse"] = backward_weights["bias_hh"]

    for block_index in range(module_config.num_fc_blocks):
        tf_prefix = f"Hidden_FC_{block_index}"
        prefix = f"classifier.{block_index}"

        state_dict[f"{prefix}.dense.weight"] = _convert_fc_weights(
            reader.get_tensor(f"{tf_prefix}/fully_connected/weights")
        )
        state_dict[f"{prefix}.dense.bias"] = torch.from_numpy(reader.get_tensor(f"{tf_prefix}/fully_connected/biases"))
        state_dict[f"{prefix}.layer_norm.weight"] = torch.from_numpy(reader.get_tensor(f"{tf_prefix}/LayerNorm/gamma"))
        state_dict[f"{prefix}.layer_norm.bias"] = torch.from_numpy(reader.get_tensor(f"{tf_prefix}/LayerNorm/beta"))

    state_dict["prediction.weight"] = _convert_fc_weights(reader.get_tensor("output_FC/fully_connected/weights"))
    state_dict["prediction.bias"] = torch.from_numpy(reader.get_tensor("output_FC/fully_connected/biases"))

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
    input_hidden_weights = tf_kernel[:input_size, :]
    hidden_hidden_weights = tf_kernel[input_size:, :]

    def _reorder_gates(weights: np.ndarray) -> np.ndarray:
        input_gate, new_gate, forget_gate, output_gate = np.split(weights, 4, axis=-1)
        return np.concatenate([input_gate, forget_gate, new_gate, output_gate], axis=-1)

    # TF BasicLSTMCell adds forget_bias at runtime, so fold it into the stored bias here.
    bias_with_forget = tf_bias.copy()
    bias_with_forget[2 * hidden_size : 3 * hidden_size] += 1.0

    return {
        "weight_ih": torch.from_numpy(_reorder_gates(input_hidden_weights).T.copy()),
        "weight_hh": torch.from_numpy(_reorder_gates(hidden_hidden_weights).T.copy()),
        "bias_ih": torch.from_numpy(_reorder_gates(bias_with_forget).copy()),
        "bias_hh": torch.zeros(4 * hidden_size),
    }


original_vocab_list = ["A", "U", "C", "G"]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
