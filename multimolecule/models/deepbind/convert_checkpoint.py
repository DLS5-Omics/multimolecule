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

import json
import os
from collections import OrderedDict
from typing import Any, Dict

import chanfig
import torch

from multimolecule.models import DeepBindConfig as Config
from multimolecule.models import DeepBindForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings as convert_dna_word_embeddings
from multimolecule.tokenisers.dna.utils import get_alphabet as get_dna_alphabet
from multimolecule.tokenisers.dna.utils import get_tokenizer_config as get_dna_tokenizer_config
from multimolecule.tokenisers.rna.utils import convert_word_embeddings as convert_rna_word_embeddings
from multimolecule.tokenisers.rna.utils import get_alphabet as get_rna_alphabet
from multimolecule.tokenisers.rna.utils import get_tokenizer_config as get_rna_tokenizer_config

torch.manual_seed(1016)


# Upstream DeepBind (Alipanahi et al., Nat Biotechnol 2015) one-hot encodes nucleotides as
# ["A", "C", "G", "T"] for both DNA and RNA models (RNA uracil shares the T channel). The Kipoi
# distribution (jisraeli/DeepBind via https://kipoi.org/models/DeepBind/) ships each per-protein
# checkpoint as a Keras `.json` architecture + `.h5` weights pair pinned to the original
# h5py-2.x serialisation; the converter below reads either the Kipoi pair or a raw `.h5`
# previously deserialised with TensorFlow / Keras.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

CONV_KEY = "model.encoder.conv"
DENSE_KEY = "model.encoder.dense"
DECODER_KEY = "sequence_head.decoder"


def convert_checkpoint(convert_config):
    print(f"Converting DeepBind checkpoint at {convert_config.checkpoint_path}")
    config = Config(
        molecule=convert_config.molecule,
        num_filters=convert_config.num_filters,
        kernel_size=convert_config.kernel_size,
        pooling=convert_config.pooling,
        num_hidden=convert_config.num_hidden,
    )
    model = Model(config)

    if config.molecule == "dna":
        alphabet = get_dna_alphabet("nucleobase", prepend_tokens=[])
        tokenizer_config = chanfig.NestedDict(get_dna_tokenizer_config())
        convert_word_embeddings = convert_dna_word_embeddings
    else:
        alphabet = get_rna_alphabet("nucleobase", prepend_tokens=[])
        tokenizer_config = chanfig.NestedDict(get_rna_tokenizer_config())
        convert_word_embeddings = convert_rna_word_embeddings
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    weights = _load_original_weights(convert_config.checkpoint_path, convert_config.arch_path)
    state_dict = _convert_checkpoint(weights, config)
    key = f"{CONV_KEY}.weight"
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


def _load_original_weights(checkpoint_path: str, arch_path: str | None) -> Dict[str, Any]:
    """Load the upstream DeepBind Keras checkpoint and return a dict of named NumPy arrays.

    The Kipoi DeepBind distribution ships per-protein Keras model pairs (`*.json` architecture +
    `*.h5` weights). The architecture controls the layer order; the weights file stores the
    matrices indexed by Keras layer name. Both files are pinned to the h5py-2.x serialisation
    format used by the original release. Pass the weights `.h5` via `--checkpoint_path` and the
    architecture `.json` via `--arch_path` (optional; if omitted the converter reads the layer
    order directly from the `.h5`).
    """
    if not (checkpoint_path and os.path.isfile(checkpoint_path)):
        raise FileNotFoundError(
            "Upstream DeepBind weights file not found. Download a per-protein Kipoi DeepBind "
            "checkpoint (see https://kipoi.org/models/DeepBind/ and the `models.tsv` URLs in "
            "`https://github.com/kipoi/models/tree/master/DeepBind`) and pass the resulting "
            "`.h5` via `--checkpoint_path`."
        )
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading the Keras DeepBind checkpoint requires the conversion-only dependency "
            "`h5py` (`pip install h5py`)."
        ) from error
    layers: Dict[str, Dict[str, Any]] = {}
    with h5py.File(checkpoint_path, "r") as h5:
        # Keras stores per-layer weights under `model_weights/<layer_name>/<param_name>`.
        root = h5["model_weights"] if "model_weights" in h5 else h5
        for layer_name in list(root.keys()):
            group = root[layer_name]
            params: Dict[str, Any] = {}
            for param_name in list(group.keys()):
                node = group[param_name]
                if hasattr(node, "shape"):  # leaf dataset
                    params[param_name] = node[()]
                else:  # nested group: walk one more level
                    for sub_name in list(node.keys()):
                        sub = node[sub_name]
                        if hasattr(sub, "shape"):
                            params[sub_name] = sub[()]
            if params:
                layers[layer_name] = params

    arch_layer_order = None
    if arch_path and os.path.isfile(arch_path):
        with open(arch_path) as handle:
            arch_layer_order = [layer["config"]["name"] for layer in json.load(handle)["config"]["layers"]]
    return {"layers": layers, "arch_layer_order": arch_layer_order}


def _convert_checkpoint(weights: Dict[str, Any], config: Config) -> OrderedDict:
    """Translate the Keras DeepBind weight dict into the MultiMolecule state-dict layout.

    The upstream Keras DeepBind model is a flat ``Sequential`` of
    ``Conv1D -> ReLU -> {GlobalMaxPool | concat(GlobalMaxPool, GlobalAvgPool)} ->
    [Dropout -> Dense -> ReLU] -> Dense``.
    Each parameterised Keras layer maps 1:1 onto the MultiMolecule module tree:

    * the convolution → ``model.encoder.conv``
    * the optional hidden dense → ``model.encoder.dense``
    * the final scalar dense → ``sequence_head.decoder``

    Keras `Conv1D` stores its kernel as ``(kernel, in_channels, out_channels)`` whereas
    PyTorch `nn.Conv1d` expects ``(out_channels, in_channels, kernel)``; Keras `Dense` stores
    its kernel as ``(in_features, out_features)`` whereas PyTorch `nn.Linear` expects
    ``(out_features, in_features)``. Both are transposed below.
    """
    layers: Dict[str, Dict[str, Any]] = weights["layers"]
    arch_layer_order = weights["arch_layer_order"]

    layer_order = arch_layer_order if arch_layer_order is not None else list(layers.keys())
    conv_layers = [name for name in layer_order if _has_kernel(layers.get(name)) and _is_conv(layers.get(name))]
    dense_layers = [name for name in layer_order if _has_kernel(layers.get(name)) and not _is_conv(layers.get(name))]

    if len(conv_layers) != 1:
        raise ValueError(f"Expected exactly 1 convolutional layer, found {len(conv_layers)}: {conv_layers}")
    expected_dense = 2 if config.num_hidden > 0 else 1
    if len(dense_layers) != expected_dense:
        raise ValueError(
            f"Expected {expected_dense} dense layer(s) for num_hidden={config.num_hidden}, "
            f"found {len(dense_layers)}: {dense_layers}"
        )

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    conv_params = layers[conv_layers[0]]
    conv_kernel = torch.as_tensor(_kernel(conv_params)).float()  # (kernel, in_channels, out_channels)
    state_dict[f"{CONV_KEY}.weight"] = conv_kernel.permute(2, 1, 0).contiguous()
    bias = _bias(conv_params)
    if bias is None:
        raise ValueError("DeepBind Conv1D is expected to have a bias term.")
    state_dict[f"{CONV_KEY}.bias"] = torch.as_tensor(bias).float()

    if config.num_hidden > 0:
        hidden_params = layers[dense_layers[0]]
        decoder_params = layers[dense_layers[1]]
        state_dict[f"{DENSE_KEY}.weight"] = torch.as_tensor(_kernel(hidden_params)).float().t().contiguous()
        hidden_bias = _bias(hidden_params)
        if hidden_bias is None:
            raise ValueError("DeepBind hidden Dense is expected to have a bias term.")
        state_dict[f"{DENSE_KEY}.bias"] = torch.as_tensor(hidden_bias).float()
    else:
        decoder_params = layers[dense_layers[0]]

    state_dict[f"{DECODER_KEY}.weight"] = torch.as_tensor(_kernel(decoder_params)).float().t().contiguous()
    decoder_bias = _bias(decoder_params)
    if decoder_bias is None:
        raise ValueError("DeepBind final Dense is expected to have a bias term.")
    state_dict[f"{DECODER_KEY}.bias"] = torch.as_tensor(decoder_bias).float()

    return state_dict


def _has_kernel(params: Dict[str, Any] | None) -> bool:
    if not params:
        return False
    return any(key.startswith("kernel") for key in params)


def _is_conv(params: Dict[str, Any] | None) -> bool:
    # Keras Conv1D kernels are 3-D (kernel, in_channels, out_channels); Dense kernels are 2-D.
    if not params:
        return False
    kernel = _kernel(params)
    return kernel is not None and getattr(kernel, "ndim", len(getattr(kernel, "shape", []))) == 3


def _kernel(params: Dict[str, Any]):
    for key in ("kernel:0", "kernel"):
        if key in params:
            return params[key]
    return None


def _bias(params: Dict[str, Any]):
    for key in ("bias:0", "bias"):
        if key in params:
            return params[key]
    return None


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""
    arch_path: str = ""
    molecule: str = "dna"
    num_filters: int = 16
    kernel_size: int = 24
    pooling: str = "maxavg"
    num_hidden: int = 32


if __name__ == "__main__":
    # TODO: fetch upstream weights. The DeepBind Kipoi distribution previously hosted per-protein
    # checkpoints at `https://sandbox.zenodo.org/record/248887/files/<protein>.h5`, but the
    # sandbox record is currently unavailable (HTTP 404). Once a stable mirror of the Kipoi
    # DeepBind weights is identified (see https://github.com/kipoi/models/issues/65 and the
    # `models.tsv` table at https://github.com/kipoi/models/tree/master/DeepBind), pass the
    # downloaded `.h5` via `--checkpoint_path` (and optionally `--arch_path` for the matching
    # `.json` architecture file) and rerun this script.
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
