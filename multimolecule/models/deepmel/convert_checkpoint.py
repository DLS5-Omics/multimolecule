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

from multimolecule.models import DeepMelConfig as Config
from multimolecule.models import DeepMelForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream DeepMEL (aertslab/DeepMEL, Taskiran, Minnoye & Aerts, Genome Research 2020) one-hot encodes DNA in the
# order ["A", "C", "G", "T"]. The released checkpoint is `DeepMEL.hdf5` from Zenodo record 3592129, a Keras 2.1.5
# functional model with the layers conv1d_1 / time_distributed_1 / bidirectional_1 (LSTM) / dense_2 / dense_3.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    print(f"Converting DeepMEL checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.input_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    root = convert_config.checkpoint_path
    if not (root and os.path.isfile(root)):
        raise FileNotFoundError(
            "No upstream DeepMEL checkpoint found. Download `DeepMEL.hdf5` from Zenodo record 3592129 "
            "(https://zenodo.org/records/3592129) and pass it via `--checkpoint_path`."
        )
    state_dict = _convert_checkpoint(root, config)

    conv_key = "model.encoder.conv.weight"
    weight = state_dict.get(conv_key)
    if weight is not None:
        # Convert the upstream `["A", "C", "G", "T"]` channels into the MultiMolecule DNA alphabet
        # (`"ACGTN..."`); the extra "N" / IUPAC slots are zero-initialised.
        converted = convert_one_hot_embeddings(
            weight,
            old_vocab=ORIGINAL_VOCAB_LIST,
            new_vocab=new_vocab_list,
            convert_word_embeddings=convert_word_embeddings,
        )
        # Upstream DeepMEL leaves N/ambiguous bases as all-zero one-hot rows. The shared word-embedding converter
        # averages ambiguous token weights, which is useful for learned embeddings but wrong for one-hot CNN inputs.
        for index, token in enumerate(new_vocab_list):
            if token not in ORIGINAL_VOCAB_LIST:
                converted[:, index, :] = 0
        state_dict[conv_key] = converted

    reference_state = model.state_dict()
    for key, value in reference_state.items():
        if key not in state_dict and key.endswith("num_batches_tracked"):
            state_dict[key] = value
    load_checkpoint(model, state_dict)

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _read_layer_weights(h5file, layer_name: str) -> dict:
    """Read every weight tensor stored under a Keras layer in the `model_weights/<layer>/...` HDF5 hierarchy."""
    import numpy as np  # noqa: PLC0415

    group = h5file["model_weights"][layer_name] if "model_weights" in h5file else h5file[layer_name]
    weights: dict[str, np.ndarray] = {}

    def _collect(node):
        import h5py  # noqa: PLC0415

        for key in node.keys():
            item = node[key]
            if isinstance(item, h5py.Group):
                _collect(item)
            else:
                short = key.split(":")[0]
                weights[short] = item[()]

    _collect(group)
    return weights


def _lstm_layer_weights(h5file, layer_name: str) -> tuple[dict, dict]:
    """Read the forward + backward LSTM kernels of an upstream `Bidirectional` Keras layer."""
    import numpy as np  # noqa: PLC0415

    group = h5file["model_weights"][layer_name][layer_name]
    forward: dict[str, np.ndarray] = {}
    backward: dict[str, np.ndarray] = {}
    for name, sub in group.items():
        target = forward if name.startswith("forward") else backward if name.startswith("backward") else None
        if target is None:
            continue
        for tensor_name in sub.keys():
            short = tensor_name.split(":")[0]
            target[short] = sub[tensor_name][()]
    return forward, backward


def _convert_checkpoint(file: str, config: Config) -> OrderedDict:
    import h5py  # noqa: PLC0415

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as h5file:
        # Conv1D: Keras (kernel_size, in_channels, out_channels) -> PyTorch (out_channels, in_channels, kernel_size).
        conv = _read_layer_weights(h5file, "conv1d_1")
        kernel = torch.from_numpy(conv["kernel"]).permute(2, 1, 0).contiguous()
        state_dict["model.encoder.conv.weight"] = kernel
        state_dict["model.encoder.conv.bias"] = torch.from_numpy(conv["bias"])

        # TimeDistributed(Dense): Keras (in, out) -> PyTorch (out, in).
        td = _read_layer_weights(h5file, "time_distributed_1")
        state_dict["model.encoder.time_distributed.weight"] = (
            torch.from_numpy(td["kernel"]).transpose(0, 1).contiguous()
        )
        state_dict["model.encoder.time_distributed.bias"] = torch.from_numpy(td["bias"])

        # Bidirectional(LSTM): Keras stores kernel `(input_size, 4 * hidden)` and recurrent_kernel `(hidden,
        # 4 * hidden)` in gate order `[i, f, c, o]`. The MultiMolecule `DeepMelLstm` mirrors that gate order so the
        # conversion only requires a transpose, with no gate-permutation needed.
        forward_lstm, backward_lstm = _lstm_layer_weights(h5file, "bidirectional_1")
        _set_lstm(state_dict, "model.encoder.lstm.forward_lstm", forward_lstm)
        _set_lstm(state_dict, "model.encoder.lstm.backward_lstm", backward_lstm)

        # Dense_2 -> per-branch fully-connected layer feeding the prediction head.
        dense_2 = _read_layer_weights(h5file, "dense_2")
        state_dict["model.encoder.fc.weight"] = torch.from_numpy(dense_2["kernel"]).transpose(0, 1).contiguous()
        state_dict["model.encoder.fc.bias"] = torch.from_numpy(dense_2["bias"])

        # Dense_3 -> final 24-way decoder, mapped onto the `SequencePredictionHead.decoder` linear.
        dense_3 = _read_layer_weights(h5file, "dense_3")
        state_dict["sequence_head.decoder.weight"] = torch.from_numpy(dense_3["kernel"]).transpose(0, 1).contiguous()
        state_dict["sequence_head.decoder.bias"] = torch.from_numpy(dense_3["bias"])

    return state_dict


def _set_lstm(state_dict: OrderedDict, prefix: str, weights: dict) -> None:
    """Translate one Keras LSTM into the corresponding `DeepMelLstm` parameters."""
    state_dict[f"{prefix}.weight_ih"] = torch.from_numpy(weights["kernel"]).transpose(0, 1).contiguous()
    state_dict[f"{prefix}.weight_hh"] = torch.from_numpy(weights["recurrent_kernel"]).transpose(0, 1).contiguous()
    state_dict[f"{prefix}.bias"] = torch.from_numpy(weights["bias"])


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
