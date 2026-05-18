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

from multimolecule.models import MtSpliceConfig as Config
from multimolecule.models import MtSpliceModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Upstream MTSplice ("deep" variant, the package default) is an ensemble of four
# Keras models `mtsplice_deep{0,1,2,3}.h5`. The ensemble averaging is an implementation detail;
# the default checkpoint uses one member (`mtsplice_deep0.h5`).
DEFAULT_CHECKPOINT = "mtsplice_deep0.h5"

# Upstream one-hot channel order (mmsplice.layers.DNA).
UPSTREAM_DNA = ["A", "C", "G", "T"]

# Keras layer name -> MultiMolecule sub-path. The left tower (input `seql`)
# scores the acceptor region, the right tower (input `seqr`) scores the donor
# region. Keras Conv1D kernels have shape (kernel, in, out); torch Conv1d expects
# (out, in, kernel), so conv kernels are permuted (2, 1, 0). Keras Dense kernels
# have shape (in, out); torch Linear expects (out, in), so dense kernels are
# transposed. SplineWeight1D kernels are (n_bases, channels) in both frameworks
# and are kept as-is.
NAME_MAPPING = {
    "conv1d_1": "acceptor_tower.stem",
    "conv1d_10": "donor_tower.stem",
    "bn5": "prediction.norm",
    "bn6": "prediction.post_norm",
    "dense2": "prediction.decoder",
    "dense": "prediction.dense",
    "splinel": "acceptor_tower.spline",
    "spliner": "donor_tower.spline",
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}

# Residual blocks: Keras `batch_normalization_{i}` / `conv1d_{i+1}` map to the
# left tower; `batch_normalization_{i+8}` / `conv1d_{i+10}` map to the right
# tower. Block `b` (0-indexed) has `norm` before `conv`.
for _b in range(8):
    NAME_MAPPING[f"batch_normalization_{_b + 1}"] = f"acceptor_tower.blocks.{_b}.norm"
    NAME_MAPPING[f"conv1d_{_b + 2}"] = f"acceptor_tower.blocks.{_b}.conv"
    NAME_MAPPING[f"batch_normalization_{_b + 9}"] = f"donor_tower.blocks.{_b}.norm"
    NAME_MAPPING[f"conv1d_{_b + 11}"] = f"donor_tower.blocks.{_b}.conv"


def convert_checkpoint(convert_config):
    print(f"Converting MTSplice checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    config.architectures = ["MtSpliceModel"]

    model = Model(config)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    new_vocab_list = list(alphabet.vocabulary)

    path = convert_config.checkpoint_path
    if os.path.isdir(path):
        path = os.path.join(path, DEFAULT_CHECKPOINT)
    state_dict = _convert_checkpoint(path, model, new_vocab_list)
    load_checkpoint(model, state_dict)

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(file, model, new_vocab_list: list[str]):
    import h5py  # noqa: PLC0415  optional, conversion-only dependency

    target_keys = set(model.state_dict().keys())
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(file, "r") as h5:
        weights = h5["model_weights"]
        layer_names = [_decode(n) for n in weights.attrs["layer_names"]]
        for layer_name in layer_names:
            group = weights[layer_name]
            weight_names = [_decode(n) for n in group.attrs.get("weight_names", [])]
            for weight_name in weight_names:
                dataset = group
                for part in weight_name.split("/"):
                    dataset = dataset[part]
                value = torch.from_numpy(dataset[()].astype("float32"))
                short = weight_name.split("/")[-1].split(":")[0]
                new_name = _convert_name(layer_name, short)
                if "kernel" in weight_name and value.dim() == 3:
                    value = value.permute(2, 1, 0).contiguous()
                    if new_name == "acceptor_tower.stem.weight" or new_name == "donor_tower.stem.weight":
                        value = convert_one_hot_embeddings(
                            value,
                            old_vocab=UPSTREAM_DNA,
                            new_vocab=new_vocab_list,
                            convert_word_embeddings=convert_word_embeddings,
                        )
                elif "kernel" in weight_name and value.dim() == 2 and "spline" not in layer_name:
                    value = value.t().contiguous()
                if new_name not in target_keys:
                    raise KeyError(
                        f"Converted key '{new_name}' (from Keras '{layer_name}/{weight_name}') "
                        f"is absent from the MTSplice model state dict."
                    )
                state_dict[new_name] = value
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value
    return state_dict


def _convert_name(layer_name: str, weight_short: str) -> str:
    module = NAME_MAPPING.get(layer_name)
    if module is None:
        raise KeyError(f"Unknown Keras layer '{layer_name}' in MTSplice checkpoint.")
    if "spline" in layer_name:
        # SplineWeight1D stores its (n_bases, channels) coefficients as `kernel`.
        return f"{module}.kernel"
    suffix = NAME_MAPPING.get(weight_short, weight_short)
    return f"{module}.{suffix}"


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
