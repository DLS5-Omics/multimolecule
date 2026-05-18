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

# The upstream MMSplice weights are distributed by the original authors
# (Cheng et al. 2019, Genome Biology) in the `mmsplice` PyPI package and the
# gagneurlab/MMSplice_MTSplice repository, one Keras `.h5` file per module.
# `tensorflow` / `h5py` are *conversion-only* dependencies; the converted
# MultiMolecule model is pure-torch and never imports them at runtime.


from __future__ import annotations

import os
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import MmSpliceConfig as Config
from multimolecule.models import MmSpliceForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Upstream MMSplice ships one Keras `.h5` file per module.
# Map the upstream module filename (without extension) to the MultiMolecule module name.
MODULE_FILES = {
    "Intron3": "acceptor_intron",
    "Acceptor": "acceptor",
    "Exon": "exon",
    "Donor": "donor",
    "Intron5": "donor_intron",
}

# Per-module mapping from the upstream Keras layer name to the MultiMolecule
# sub-path inside a `MmSpliceModule`. Keras Conv1D kernels have shape
# (kernel, in, out); torch Conv1d expects (out, in, kernel) so conv kernels are
# permuted (2, 1, 0). Keras Dense kernels have shape (in, out); torch Linear
# expects (out, in) so dense kernels are transposed. Keras flattens in
# (length, channels) order, identical to our `transpose(1, 2).reshape`, so the
# flattened Dense kernel transposes without any further reordering.
LAYER_MAPPING = {
    "acceptor_intron": {
        "conv": "network.conv",
        "dense3": "network.decoder",
    },
    "acceptor": {
        "convSpliceSite": "network.conv",
        "batch_normalization_1": "network.conv_norm",
        "conv1by1": "network.pointwise",
        "batch_normalization_2": "network.pointwise_norm",
        "denseSS": "network.decoder",
    },
    "exon": {
        "conv": "network.conv",
        "batch_normalization_1": "network.norm",
        "dense3": "network.decoder",
    },
    "donor": {
        "dense1": "network.blocks.0.dense",
        "batch_normalization_1": "network.blocks.0.norm",
        "dense2": "network.blocks.1.dense",
        "batch_normalization_2": "network.blocks.1.norm",
        "dense_1": "network.decoder",
    },
    "donor_intron": {
        "conv": "network.conv",
        "dense5": "network.decoder",
    },
}

WEIGHT_MAPPING = {
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}

ORIGINAL_VOCAB_LIST = ["A", "C", "G", "U"]


def convert_checkpoint(convert_config):
    print(f"Converting MMSplice checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    config.architectures = ["MmSpliceForSequencePrediction"]

    model = Model(config)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    new_vocab_list = list(alphabet.vocabulary)

    root = convert_config.checkpoint_path
    for filename, module_name in MODULE_FILES.items():
        path = os.path.join(root, filename + ".h5")
        state_dict = _convert_module(path, module_name, new_vocab_list)
        module = model.model.region_models[module_name]
        module_state = module.state_dict()
        for key, value in module_state.items():
            if key.endswith("num_batches_tracked") and key not in state_dict:
                state_dict[key] = value
        load_checkpoint(module, state_dict)

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_module(file: str, module_name: str, new_vocab_list: list[str]) -> OrderedDict[str, torch.Tensor]:
    # Read the Keras weights straight from the HDF5 archive. This avoids
    # reconstructing the upstream functional graph (which uses custom layers
    # such as `ConvDNA` / `GlobalAveragePooling1D_Mask0`) and keeps the
    # conversion dependency limited to `h5py` plus `numpy`/`torch`.
    import h5py  # noqa: PLC0415  conversion-only dependency
    import numpy as np  # noqa: PLC0415  conversion-only dependency

    layer_mapping = LAYER_MAPPING[module_name]
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as h5:
        weights = h5["model_weights"]
        for layer_name, prefix in layer_mapping.items():
            group = weights[layer_name]
            weight_names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs["weight_names"]]
            for weight_name in weight_names:
                dataset = group
                for part in weight_name.split("/"):
                    dataset = dataset[part]
                name = weight_name.split("/")[-1].split(":")[0]
                torch_name = WEIGHT_MAPPING[name]
                tensor = torch.from_numpy(np.asarray(dataset))
                if name == "kernel" and tensor.dim() == 3:
                    tensor = tensor.permute(2, 1, 0).contiguous()
                    if prefix == "network.conv":
                        tensor = convert_one_hot_embeddings(
                            tensor,
                            old_vocab=ORIGINAL_VOCAB_LIST,
                            new_vocab=new_vocab_list,
                            convert_word_embeddings=convert_word_embeddings,
                        )
                elif name == "kernel" and tensor.dim() == 2:
                    tensor = tensor.t().contiguous()
                state_dict[f"{prefix}.{torch_name}"] = tensor
    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
