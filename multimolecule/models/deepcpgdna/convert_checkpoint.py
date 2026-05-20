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
from collections.abc import Iterable
from pathlib import Path

import chanfig
import h5py
import torch

from multimolecule.models import DeepCpgDnaConfig as Config
from multimolecule.models import DeepCpgDnaForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream DeepCpG (Angermueller et al. 2017, cangermueller/deepcpg) one-hot encodes DNA as ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Upstream Keras 1.1.2 `model_weights.h5` file from Zenodo record 1466079 (DeepCpG model zoo). Each variant is one
# pretrained DNA submodule (e.g. Smallwood2014_serum_dna with 18 single cells); the cells (and therefore `num_labels`)
# are dataset-specific.
DEFAULT_VARIANT = "deepcpgdna-smallwood2014-serum"
VARIANTS = {
    "smallwood2014-serum": ("deepcpgdna-smallwood2014-serum", "Smallwood2014_serum_dna-model"),
    "smallwood2014-2i": ("deepcpgdna-smallwood2014-2i", "Smallwood2014_2i_dna-model"),
    "hou2016-hcc": ("deepcpgdna-hou2016-hcc", "Hou2016_HCC_dna-model"),
    "hou2016-hepg2": ("deepcpgdna-hou2016-hepg2", "Hou2016_HepG2_dna-model"),
    "hou2016-mesc": ("deepcpgdna-hou2016-mesc", "Hou2016_mESC_dna-model"),
}


def convert_checkpoint(convert_config):
    print(f"Converting DeepCpG-DNA checkpoint at {convert_config.checkpoint_path}")
    config, cell_names = _infer_config(convert_config.checkpoint_path, convert_config.model_path)
    config.id2label = dict(enumerate(cell_names))
    config.label2id = {name: index for index, name in enumerate(cell_names)}
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, config, cell_names)
    key = "model.encoder.layers.0.conv.weight"
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


def _find_variant(variant: str) -> tuple[str, str]:
    normalized = variant.lower().replace("_", "-")
    if normalized in VARIANTS:
        return VARIANTS[normalized]
    for output_path, model_stem in VARIANTS.values():
        if normalized == output_path or normalized == model_stem.lower().replace("_", "-"):
            return output_path, model_stem
    variants = ", ".join(sorted(VARIANTS))
    raise ValueError(f"Unknown DeepCpG-DNA variant {variant!r}. Expected one of: {variants}.")


def _variant_paths(checkpoint_root: str, variant: str) -> tuple[str, str, str]:
    output_path, model_stem = _find_variant(variant)
    root = Path(checkpoint_root)
    model_path = root / model_stem
    checkpoint_path = root / f"{model_stem}_weights.h5"
    return output_path, str(model_path), str(checkpoint_path)


def convert_all_checkpoints(convert_config):
    missing = []
    for variant in VARIANTS:
        _, model_path, checkpoint_path = _variant_paths(convert_config.checkpoint_root, variant)
        if not os.path.exists(model_path):
            missing.append(model_path)
        if not os.path.exists(checkpoint_path):
            missing.append(checkpoint_path)
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing DeepCpG-DNA checkpoints:\n{missing_list}")

    if convert_config.repo_id is not None:
        raise ValueError(
            "Do not pass repo_id with convert_all; each variant writes to its own multimolecule/<variant> repo."
        )

    for variant in VARIANTS:
        output_path, model_path, checkpoint_path = _variant_paths(convert_config.checkpoint_root, variant)
        child = ConvertConfig()
        child.root = convert_config.root
        child.output_path = output_path
        child.checkpoint_root = convert_config.checkpoint_root
        child.checkpoint_path = checkpoint_path
        child.model_path = model_path
        child.variant = variant
        child.convert_all = False
        child.default_variant = convert_config.default_variant
        child.push_to_hub = convert_config.push_to_hub
        child.delete_existing = convert_config.delete_existing
        child.token = convert_config.token
        child.repo_id = f"multimolecule/{output_path}"
        convert_checkpoint(child)


def _infer_config(checkpoint_path: str, model_path: str | None = None) -> tuple[Config, list[str]]:
    if model_path:
        return _infer_config_from_model(model_path)
    return _infer_config_from_weights(checkpoint_path)


def _infer_config_from_model(model_path: str) -> tuple[Config, list[str]]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Upstream DeepCpG-DNA model JSON not found. Download `<variant>_dna-model` from Zenodo record 1466079 "
            "(https://zenodo.org/record/1466079) and pass it via `--model_path`, or use `--checkpoint_root` with "
            "`--variant` / `--convert_all`."
        )
    with open(model_path) as f:
        model_spec = json.load(f)

    layers = model_spec["config"]["layers"]
    input_layer = next(layer for layer in layers if layer["class_name"] == "InputLayer")
    sequence_length = input_layer["config"]["batch_input_shape"][1]
    conv_layers = [layer for layer in layers if layer["class_name"] == "Convolution1D"]
    pool_layers = [layer for layer in layers if layer["class_name"] == "MaxPooling1D"]
    dense_layer = next(layer for layer in layers if layer["name"] == "dna/dense_1")
    dropout_layer = next(layer for layer in layers if layer["class_name"] == "Dropout")
    output_layers = model_spec["config"]["output_layers"]

    config = Config(
        sequence_length=sequence_length,
        conv_channels=[layer["config"]["nb_filter"] for layer in conv_layers],
        conv_kernel_sizes=[layer["config"]["filter_length"] for layer in conv_layers],
        conv_pool_sizes=[layer["config"]["pool_length"] for layer in pool_layers],
        bottleneck_size=dense_layer["config"]["output_dim"],
        hidden_dropout=dropout_layer["config"]["p"],
        num_labels=len(output_layers),
    )
    cell_names = [name.split("/", 1)[1] if "/" in name else name for name, _, _ in output_layers]
    return config, cell_names


def _infer_cells(checkpoint_path: str) -> tuple[int, list[str]]:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Upstream DeepCpG-DNA checkpoint not found. Download `<variant>_dna-model_weights.h5` from Zenodo "
            "record 1466079 (https://zenodo.org/record/1466079) and pass it via `--checkpoint_path`."
        )
    with h5py.File(checkpoint_path, "r") as f:
        cpg_group = f["model_weights"]["cpg"]
        cell_names = sorted(cpg_group.keys())
    return len(cell_names), cell_names


def _infer_config_from_weights(checkpoint_path: str) -> tuple[Config, list[str]]:
    num_labels, cell_names = _infer_cells(checkpoint_path)
    with h5py.File(checkpoint_path, "r") as f:
        dna = f["model_weights"]["dna"]
        conv_names = _sorted_layer_names(name for name in dna.keys() if name.startswith("convolution1d_"))
        conv_channels = []
        conv_kernel_sizes = []
        for name in conv_names:
            weight = dna[name][f"{name}_W:0"]
            conv_kernel_sizes.append(int(weight.shape[0]))
            conv_channels.append(int(weight.shape[-1]))
        bottleneck_size = int(dna["dense_1"]["dense_1_b:0"].shape[0])

    if len(conv_channels) == 2:
        conv_pool_sizes = [4, 2]
    elif len(conv_channels) == 3:
        conv_pool_sizes = [4, 2, 2]
    else:
        raise ValueError(
            f"Unsupported DeepCpG-DNA checkpoint with {len(conv_channels)} convolution layers. "
            "Pass the matching `<variant>_dna-model` JSON via `--model_path` if this architecture is supported."
        )
    return (
        Config(
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_pool_sizes=conv_pool_sizes,
            bottleneck_size=bottleneck_size,
            num_labels=num_labels,
        ),
        cell_names,
    )


def _sorted_layer_names(names: Iterable[str]) -> list[str]:
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _convert_checkpoint(checkpoint_path: str, config: Config, cell_names: list[str]) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(checkpoint_path, "r") as f:
        dna = f["model_weights"]["dna"]
        # Keras 1.1.2 stored `Convolution1D` weights with shape (kernel, 1, in_channels, out_channels); the singleton
        # row dimension is collapsed to obtain a torch `Conv1d` weight of shape (out, in, kernel).
        for index in range(len(config.conv_channels)):
            name = f"convolution1d_{index + 1}"
            weight = torch.from_numpy(dna[name][f"{name}_W:0"][()]).squeeze(1).permute(2, 1, 0)
            bias = torch.from_numpy(dna[name][f"{name}_b:0"][()])
            state_dict[f"model.encoder.layers.{index}.conv.weight"] = weight.contiguous()
            state_dict[f"model.encoder.layers.{index}.conv.bias"] = bias.contiguous()

        # Keras Dense kernel: (in, out) -> torch Linear (out, in). The flattened input order is Keras `(length,
        # channels)` row-major. The MM model transposes `(channels, length) -> (length, channels)` before flattening,
        # so this column ordering already matches and only the standard transpose is required.
        dense_w = torch.from_numpy(dna["dense_1"]["dense_1_W:0"][()]).t().contiguous()
        dense_b = torch.from_numpy(dna["dense_1"]["dense_1_b:0"][()])
        state_dict["model.encoder.bottleneck.dense.weight"] = dense_w
        state_dict["model.encoder.bottleneck.dense.bias"] = dense_b

        # Per-cell decoder: each upstream cell is its own Dense(1) layer. Concatenate them in `cell_names` order so
        # `id2label[index]` matches the row in the shared SequencePredictionHead decoder.
        rows = []
        biases = []
        cpg_group = f["model_weights"]["cpg"]
        for name in cell_names:
            cell_group = cpg_group[name]["cpg"]
            w = torch.from_numpy(cell_group[f"{name}_W:0"][()])  # shape (128, 1)
            b = torch.from_numpy(cell_group[f"{name}_b:0"][()])  # shape (1,)
            rows.append(w.t())  # (1, 128)
            biases.append(b)
        state_dict["sequence_head.decoder.weight"] = torch.cat(rows, dim=0).contiguous()
        state_dict["sequence_head.decoder.bias"] = torch.cat(biases, dim=0).contiguous()

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]
    checkpoint_path: str = ""
    model_path: str | None = None
    checkpoint_root: str = "models"
    variant: str = "smallwood2014-serum"
    convert_all: bool = False
    default_variant: str | None = DEFAULT_VARIANT

    def post(self):
        if self.convert_all:
            return
        if not self.checkpoint_path:
            output_path, model_path, checkpoint_path = _variant_paths(self.checkpoint_root, self.variant)
            if self.output_path is None:
                self.output_path = output_path
            self.model_path = self.model_path or model_path
            self.checkpoint_path = checkpoint_path
        elif self.output_path is None:
            self.output_path = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    if config.convert_all:
        convert_all_checkpoints(config)
    else:
        convert_checkpoint(config)
