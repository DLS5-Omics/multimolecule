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

from multimolecule.models import A2zChromatinConfig as Config
from multimolecule.models import A2zChromatinForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream a2z-chromatin (twrightsman/a2z-regulatory, Wrightsman et al., The Plant Genome 2022) one-hot
# encodes DNA as four ["A", "C", "G", "T"] channels and represents IUPAC ambiguity codes as fractional mixtures.
# The released Kipoi checkpoints (`a2z-accessibility` -> `model-accessibility-full.h5`, `a2z-methylation` ->
# `model-methylation-full.h5`) are Keras 2.6 HDF5 weight files hosted at https://zenodo.org/record/5724562.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    print(f"Converting a2z-chromatin checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("iupac", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "."
    tokenizer_config["replace_U_with_T"] = False

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, config)
    key = "model.encoder.conv.weight"
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


def _model_weights_root(f):
    # Keras 2.6 full-model `.h5` files (saved via `model.save` / `model.save_weights(... save_format='h5')`)
    # group all layer weights under `model_weights/`; the standalone weights file used by Keras's earlier
    # `save_weights` legacy path stores them at the root. Support both by returning whichever group exists.
    return f["model_weights"] if "model_weights" in f else f


def _read_weight(f, layer: str, name: str) -> torch.Tensor:
    # Keras 2.6 `model.save_weights` stores each layer's tensors under <root>/<layer>/<layer>/<name>:0.
    root = _model_weights_root(f)
    key = f"{layer}/{layer}/{name}:0"
    if key not in root:
        raise KeyError(f"Expected weight {key} not found in a2z-chromatin checkpoint; refusing to skip conversion.")
    return torch.from_numpy(root[key][()])


def _read_bidirectional_lstm_weight(f, wrapper: str, direction: str, name: str) -> torch.Tensor:
    # `Bidirectional(LSTM)` is saved with a nested layout: <root>/<wrapper>/<wrapper>/forward_lstm/<cell>/<name>:0
    # (and the same for backward). The inner `lstm_cell_N` counter varies between checkpoints; scan to find
    # the matching forward / backward subgroup without hard-coding the suffix.
    root = _model_weights_root(f)
    wrapper_group = root[wrapper][wrapper]
    direction_group = wrapper_group[f"{direction}_lstm"]
    cell_keys = list(direction_group.keys())
    matches = [key for key in cell_keys if key.startswith("lstm_cell")]
    if not matches:
        raise KeyError(
            f"Could not locate an `lstm_cell*` group under `{wrapper}/{wrapper}/{direction}_lstm`; saw {cell_keys}."
        )
    cell = matches[0]
    dataset = direction_group[cell].get(f"{name}:0")
    if dataset is None:
        raise KeyError(
            f"Expected weight `{wrapper}/{wrapper}/{direction}_lstm/{cell}/{name}:0` not found in a2z-chromatin "
            "checkpoint; refusing to skip conversion."
        )
    return torch.from_numpy(dataset[()])


def _convert_checkpoint(checkpoint_path: str, config: Config) -> OrderedDict:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Upstream a2z-chromatin checkpoint not found. Download the released Kipoi weights "
            "(https://zenodo.org/record/5724562) and pass the `.h5` via `--checkpoint_path`. The two "
            "released variants are `model-accessibility-full.h5` and `model-methylation-full.h5`; both share "
            "the same DanQ topology and convert through this script unchanged."
        )

    try:
        import h5py  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "Reading the Keras a2z-chromatin checkpoint requires the conversion-only dependency `h5py` "
            "(`pip install h5py`)."
        ) from error

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(checkpoint_path, "r") as f:
        layer_names = _resolve_layer_names(f)

        # Convolution: Keras Conv1D kernel (kw, in, out) -> torch Conv1d (out, in, kw).
        conv_layer = layer_names["conv"]
        state_dict["model.encoder.conv.weight"] = _read_weight(f, conv_layer, "kernel").permute(2, 1, 0).contiguous()
        state_dict["model.encoder.conv.bias"] = _read_weight(f, conv_layer, "bias")

        # Bidirectional LSTM: Keras stores `kernel`, `recurrent_kernel`, and `bias` for each direction. The
        # kernel concatenates the four gates along the last axis in Keras order [i, f, c, o]; PyTorch's
        # `nn.LSTM` uses [i, f, g, o] which matches Keras's [i, f, c, o] (the cell gate is named differently
        # but occupies the same slot). PyTorch transposes the kernel relative to Keras.
        bilstm_layer = layer_names["bilstm"]
        for direction, suffix in (("forward", "l0"), ("backward", "l0_reverse")):
            kernel = _read_bidirectional_lstm_weight(f, bilstm_layer, direction, "kernel")  # (in, 4 * hidden)
            recurrent = _read_bidirectional_lstm_weight(
                f, bilstm_layer, direction, "recurrent_kernel"
            )  # (hidden, 4 * hidden)
            bias = _read_bidirectional_lstm_weight(f, bilstm_layer, direction, "bias")  # (4 * hidden,)
            state_dict[f"model.encoder.lstm.weight_ih_{suffix}"] = kernel.t().contiguous()
            state_dict[f"model.encoder.lstm.weight_hh_{suffix}"] = recurrent.t().contiguous()
            # Keras stores a single bias tensor; PyTorch splits it into `bias_ih` and `bias_hh` and sums them
            # at runtime. Put the entire bias into `bias_ih` and zero out `bias_hh` so the sum reproduces the
            # Keras-side single bias exactly.
            state_dict[f"model.encoder.lstm.bias_ih_{suffix}"] = bias
            state_dict[f"model.encoder.lstm.bias_hh_{suffix}"] = torch.zeros_like(bias)

        # Dense bottleneck: Keras Dense kernel (in, out) -> torch Linear (out, in).
        dense_layer = layer_names["dense"]
        state_dict["model.encoder.dense.weight"] = _read_weight(f, dense_layer, "kernel").t().contiguous()
        state_dict["model.encoder.dense.bias"] = _read_weight(f, dense_layer, "bias")

        # Final per-window classification cell: Dense(1, sigmoid) -> shared SequencePredictionHead decoder.
        head_layer = layer_names["head"]
        state_dict["sequence_head.decoder.weight"] = _read_weight(f, head_layer, "kernel").t().contiguous()
        state_dict["sequence_head.decoder.bias"] = _read_weight(f, head_layer, "bias")

    return state_dict


def _resolve_layer_names(f) -> dict[str, str]:
    """Find the four parameterised layer names in the upstream Keras checkpoint.

    The released `a2z-accessibility` and `a2z-methylation` checkpoints both share a four-parameterised-layer
    DanQ topology (Conv1D -> Bidirectional(LSTM) -> Dense -> Dense). Keras names them with a global counter,
    so the suffixes differ depending on how many models were instantiated in the same Python session before
    `save_weights`. Probe the standard names first, then fall back to a scan keyed on the wrapper / dense
    naming pattern.
    """
    candidates = {
        "conv": ("conv1d", "conv1d_1", "conv1d_2"),
        "bilstm": ("bidirectional", "bidirectional_1", "bidirectional_2"),
        "dense": ("dense", "dense_1", "dense_2"),
        "head": ("dense_1", "dense_2", "dense_3"),
    }
    root = _model_weights_root(f)
    resolved: dict[str, str] = {}
    for slot in ("conv", "bilstm"):
        for name in candidates[slot]:
            if name in root:
                resolved[slot] = name
                break
        if slot not in resolved:
            raise KeyError(
                f"Could not locate the {slot} layer in the a2z-chromatin checkpoint; expected one of "
                f"{candidates[slot]}."
            )
    # Resolve the two Dense layers in checkpoint order; the first is the 925-unit projection and the second
    # is the 1-unit classification cell.
    dense_candidates = list(dict.fromkeys(candidates["dense"] + candidates["head"]))
    dense_layers = [name for name in dense_candidates if name in root]
    if len(dense_layers) < 2:
        raise KeyError(
            "Could not locate the two Dense layers in the a2z-chromatin checkpoint; expected at least two "
            f"of {dense_candidates}."
        )
    resolved["dense"], resolved["head"] = dense_layers[0], dense_layers[1]
    return resolved


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
