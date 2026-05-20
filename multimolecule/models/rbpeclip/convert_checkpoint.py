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

from multimolecule.models import RbpEclipConfig as Config
from multimolecule.models import RbpEclipForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.models.rbpeclip.configuration_rbpeclip import DEFAULT_POSITION_FEATURES
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Upstream Kipoi `rbp_eclip` one-hot order is ["A", "C", "G", "U"] (concise.encodeDNA / encodeRNA).
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "U"]

# Map upstream Keras layer names to the MultiMolecule parameter prefix.  Upstream layer names follow the
# ``model()`` constructor in `Scripts/RBP/Eclip/predictive_models/model.py`:
#   conv_dna_1            -> first sequence convolution (ConvDNA, kernel_size=11)
#   conv1d_1              -> second sequence convolution (kernel_size=1)
#   batch_normalization_1 -> Keras `kl.BatchNormalization(axis=1)` after conv_dna_1
#   batch_normalization_2 -> Keras `kl.BatchNormalization(axis=1)` after conv1d_1
#   batch_normalization_3 -> Keras `kl.BatchNormalization()` over the flattened pooled sequence features
#   batch_normalization_4 -> Keras `kl.BatchNormalization()` after the hidden dense layer
#   conv_dist_<feat>      -> position-module GAM 1x1 convolution per landmark feature
#   dense_1               -> hidden FC layer that consumes the pooled sequence + position scalars
#   dense_2               -> single-output binding-score head
SEQUENCE_CONV1_KEY = "conv_dna_1"
SEQUENCE_CONV2_KEY = "conv1d_1"
SEQ_BN1_KEY = "batch_normalization_1"
SEQ_BN2_KEY = "batch_normalization_2"
POOLED_BN_KEY = "batch_normalization_3"
DENSE_BN_KEY = "batch_normalization_4"
HIDDEN_DENSE_KEY = "dense_1"
OUTPUT_DENSE_KEY = "dense_2"


def convert_checkpoint(convert_config):
    print(f"Converting RBP-eCLIP checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, config)

    # Re-order the upstream `[A, C, G, U]` channel axis to the MultiMolecule streamline-RNA order.
    key = "model.encoder.sequence_module.conv1.weight"
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError(f"Converted state dict is missing '{key}'.")
    state_dict[key] = convert_one_hot_embeddings(
        weight,
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    # BatchNorm `num_batches_tracked` is not in the upstream Keras checkpoint; copy the freshly
    # initialised model value so `load_state_dict` doesn't see a missing key.
    reference = model.state_dict()
    for key, value in reference.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(file: str, config: Config) -> OrderedDict:
    """Load a Kipoi `rbp_eclip` Keras HDF5 checkpoint and translate it into MultiMolecule state-dict keys.

    The Kipoi-distributed checkpoint is a Keras 2.1 HDF5 file. Modern Keras cannot deserialise its legacy
    graph, so the weight datasets are read directly via :mod:`h5py`. Keras stores Conv1D kernels as
    ``(kernel, in_channels, out_channels)`` and Dense kernels as ``(in, out)``; both are transposed to the
    PyTorch layout here.
    """
    if not file or not os.path.exists(file):
        raise FileNotFoundError(
            "No upstream RBP-eCLIP checkpoint found. Download one of the trained per-RBP models from "
            "the Kipoi `rbp_eclip` model group (https://kipoi.org/models/rbp_eclip/) and pass the path "
            "to its `weights.h5` via `--checkpoint_path`."
        )
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - conversion-only dependency
        raise ImportError(
            "Reading the Kipoi rbp_eclip Keras checkpoint requires the conversion-only dependency "
            "`h5py` (`pip install h5py`)."
        ) from error

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as f:
        weights = f["model_weights"]

        # Sequence module --------------------------------------------------
        # The ConvDNA layer is a Keras Conv1D with kernel shape (kernel, in_channels=4, out_channels).
        conv1_kernel, conv1_bias = _read_keras_conv(weights, SEQUENCE_CONV1_KEY)
        state_dict["model.encoder.sequence_module.conv1.weight"] = conv1_kernel
        state_dict["model.encoder.sequence_module.conv1.bias"] = conv1_bias

        conv2_kernel, conv2_bias = _read_keras_conv(weights, SEQUENCE_CONV2_KEY)
        state_dict["model.encoder.sequence_module.conv2.weight"] = conv2_kernel
        state_dict["model.encoder.sequence_module.conv2.bias"] = conv2_bias

        if config.use_batchnorm:
            for keras_key, torch_prefix in (
                (SEQ_BN1_KEY, "model.encoder.sequence_module.batch_norm1"),
                (SEQ_BN2_KEY, "model.encoder.sequence_module.batch_norm2"),
                (POOLED_BN_KEY, "model.encoder.pooled_batch_norm"),
                (DENSE_BN_KEY, "model.encoder.dense_batch_norm"),
            ):
                if keras_key in weights:
                    for suffix, value in _read_keras_batch_norm(weights, keras_key).items():
                        state_dict[f"{torch_prefix}.{suffix}"] = value

        # Position module --------------------------------------------------
        feature_names = list(config.position_feature_names) or list(DEFAULT_POSITION_FEATURES)
        for index, feat_name in enumerate(feature_names):
            keras_name = f"conv_dist_{feat_name}"
            kernel, bias = _read_keras_conv(weights, keras_name)
            # The upstream layer is a Conv1D with kernel_size=1 on the (1, n_bases) spline-basis input,
            # which is equivalent to a per-feature `Linear(n_bases, num_position_filters)`.
            # Conv1D kernel shape after `_read_keras_conv` is (out_channels, in_channels, 1); squeeze the
            # spatial dimension and keep (out_channels, in_channels) for the Linear weight layout.
            linear_weight = kernel.squeeze(-1)
            state_dict[f"model.encoder.position_module.linears.{index}.weight"] = linear_weight
            state_dict[f"model.encoder.position_module.linears.{index}.bias"] = bias

        # Hidden dense + output dense -------------------------------------
        hidden_kernel, hidden_bias = _read_keras_dense(weights, HIDDEN_DENSE_KEY)
        state_dict["model.encoder.dense.weight"] = hidden_kernel
        state_dict["model.encoder.dense.bias"] = hidden_bias

        output_kernel, output_bias = _read_keras_dense(weights, OUTPUT_DENSE_KEY)
        state_dict["sequence_head.decoder.weight"] = output_kernel
        state_dict["sequence_head.decoder.bias"] = output_bias

    return state_dict


def _read_keras_conv(weights, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    group = weights[layer_name][layer_name]
    kernel = torch.from_numpy(group["kernel:0"][()]).float()
    bias = torch.from_numpy(group["bias:0"][()]).float()
    # Keras Conv1D kernel: (kernel, in_channels, out_channels) -> torch (out_channels, in_channels, kernel).
    if kernel.dim() != 3:
        raise ValueError(
            f"Expected Keras Conv1D kernel for '{layer_name}' to have 3 dimensions, got shape {tuple(kernel.shape)}."
        )
    kernel = kernel.permute(2, 1, 0).contiguous()
    return kernel, bias


def _read_keras_batch_norm(weights, layer_name: str) -> dict[str, torch.Tensor]:
    group = weights[layer_name][layer_name]
    gamma = torch.from_numpy(group["gamma:0"][()]).float()
    beta = torch.from_numpy(group["beta:0"][()]).float()
    moving_mean = torch.from_numpy(group["moving_mean:0"][()]).float()
    moving_variance = torch.from_numpy(group["moving_variance:0"][()]).float()
    return {
        "weight": gamma,
        "bias": beta,
        "running_mean": moving_mean,
        "running_var": moving_variance,
    }


def _read_keras_dense(weights, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    group = weights[layer_name][layer_name]
    kernel = torch.from_numpy(group["kernel:0"][()]).float()
    bias = torch.from_numpy(group["bias:0"][()]).float()
    # Keras Dense kernel: (in, out) -> torch Linear (out, in).
    if kernel.dim() != 2:
        raise ValueError(
            f"Expected Keras Dense kernel for '{layer_name}' to have 2 dimensions, got shape {tuple(kernel.shape)}."
        )
    return kernel.t().contiguous(), bias


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    # TODO: fetch and convert one trained RBP per release (e.g. ``rbpeclip-hnrnpk``); the upstream
    # weights are not redistributed here. Download a Kipoi `rbp_eclip` `weights.h5` and pass it via
    # ``--checkpoint_path``.
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
