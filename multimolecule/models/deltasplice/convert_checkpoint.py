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
from pathlib import Path

import chanfig
import torch

from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.models.deltasplice.configuration_deltasplice import DeltaSpliceConfig as Config
from multimolecule.models.deltasplice.modeling_deltasplice import DeltaSpliceModel as Model
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Upstream DeltaSplice one-hot input channel order maps T to the RNA U channel in MultiMolecule.
ORIGINAL_NUCLEOTIDE_ORDER = ["A", "C", "G", "U"]
DEFAULT_ENSEMBLE_MEMBERS = [f"model.ckpt-{index}" for index in range(5)]
DEFAULT_VARIANT = "deltasplice"
VARIANT_DIRECTORIES = {
    "deltasplice": "DeltaSplice_models",
    "deltasplice-human": "DeltaSplice_human",
}


def convert_checkpoint(convert_config):
    variant = _resolve_variant(convert_config.variant)
    if convert_config.checkpoint_path is None:
        convert_config.checkpoint_path = str(Path(convert_config.checkpoint_root) / VARIANT_DIRECTORIES[variant])
    print(f"Converting DeltaSplice checkpoint at {convert_config.checkpoint_path}")

    config = Config(num_ensemble=len(DEFAULT_ENSEMBLE_MEMBERS))
    config.architectures = ["DeltaSpliceModel"]
    model = Model(config)

    checkpoint_root = Path(convert_config.checkpoint_path)
    model_alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    new_vocab = list(model_alphabet)

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for member_index, member in enumerate(DEFAULT_ENSEMBLE_MEMBERS):
        member_path = checkpoint_root / member
        if not member_path.exists():
            raise FileNotFoundError(
                "Upstream DeltaSplice weights not found: "
                f"{member_path}. Download the official chaolinzhanglab/DeltaSplice repository and pass "
                "--checkpoint_path pointing at its `deltasplice/pretrained_models/DeltaSplice_models` directory."
            )
        member_state = _convert_checkpoint(member_path, new_vocab)
        member_module_state = model.members[member_index].state_dict()
        for key, value in member_module_state.items():
            if key.endswith("num_batches_tracked") and key not in member_state:
                member_state[key] = value
        for key, value in member_state.items():
            state_dict[f"members.{member_index}.{key}"] = value

    load_checkpoint(model, state_dict)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def convert_all_checkpoints(convert_config):
    """Convert all public DeltaSplice data variants, each with the full five-member ensemble."""
    if convert_config.repo_id is not None:
        raise ValueError("Do not pass repo_id with convert_all; each variant writes to its own repo.")
    for variant in VARIANT_DIRECTORIES:
        child = ConvertConfig()
        child.root = convert_config.root
        child.output_path = variant
        child.checkpoint_path = str(Path(convert_config.checkpoint_root) / VARIANT_DIRECTORIES[variant])
        child.checkpoint_root = convert_config.checkpoint_root
        child.variant = variant
        child.convert_all = False
        child.default_variant = convert_config.default_variant
        child.push_to_hub = convert_config.push_to_hub
        child.delete_existing = convert_config.delete_existing
        child.validate_golden = convert_config.validate_golden
        child.delete_after_validate = convert_config.delete_after_validate
        child.golden_root = convert_config.golden_root
        child.token = convert_config.token
        child.repo_id = f"multimolecule/{variant}"
        convert_checkpoint(child)


def _resolve_variant(variant: str | None) -> str:
    variant = DEFAULT_VARIANT if variant is None else variant
    if variant not in VARIANT_DIRECTORIES:
        raise ValueError(f"Unsupported DeltaSplice variant {variant!r}. Expected one of {tuple(VARIANT_DIRECTORIES)}.")
    return variant


def _convert_checkpoint(file: Path, new_vocab: list) -> OrderedDict[str, torch.Tensor]:
    original = torch.load(file, map_location="cpu", weights_only=True)
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in original.items():
        new_name = convert_original_state_dict_key(name)
        if new_name is None:
            continue
        state_dict[new_name] = convert_original_state_dict_value(name, value, new_vocab)
    return state_dict


def _convert_prediction_name(prefix: str, rest: str) -> str:
    for old, new in {
        "0.": "dense.",
        "2.": "intermediate.",
        "4.": "output.",
    }.items():
        if rest.startswith(old):
            return f"{prefix}.{rest.replace(old, new, 1)}"
    raise ValueError(f"Unexpected DeltaSplice prediction-head key: {prefix}.{rest}")


def convert_original_state_dict_key(name: str) -> str | None:
    name = name.removeprefix("encode.module.")
    if name.startswith("targetconv1d."):
        return name.replace("targetconv1d.", "encoder.projection.", 1)
    if name.startswith("bn."):
        return name.replace("bn.", "encoder.norm.", 1)
    if name.startswith("encodenet."):
        _, layer_index, _, layer_item, suffix = name.split(".", 4)
        layer = {
            "0": "norm1",
            "2": "conv1",
            "3": "norm2",
            "6": "conv2",
        }.get(layer_item)
        if layer is None:
            return None
        return f"encoder.layers.{layer_index}.{layer}.{suffix}"
    if name.startswith("usagelinear."):
        rest = name.removeprefix("usagelinear.")
        if rest.startswith("0."):
            return f"reference_projection.{rest.replace('0.', 'dense.', 1)}"
        if rest.startswith("2."):
            return f"reference_projection.{rest.replace('2.', 'output.', 1)}"
        raise ValueError(f"Unexpected DeltaSplice reference-projection key: {name}")
    if name.startswith("out_usage_net."):
        return _convert_prediction_name("usage_prediction", name.removeprefix("out_usage_net."))
    if name.startswith("out_delta_net."):
        return _convert_prediction_name("delta_prediction", name.removeprefix("out_delta_net."))
    if name.startswith("out_site_net."):
        return _convert_prediction_name("site_prediction", name.removeprefix("out_site_net."))
    raise ValueError(f"Unexpected upstream DeltaSplice key: {name}")


def convert_original_state_dict_value(name: str, value: torch.Tensor, new_vocab: list) -> torch.Tensor:
    name = name.removeprefix("encode.module.")
    if name == "targetconv1d.weight":
        return convert_one_hot_embeddings(
            value,
            old_vocab=ORIGINAL_NUCLEOTIDE_ORDER,
            new_vocab=new_vocab,
            convert_word_embeddings=convert_word_embeddings,
        )
    return value


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]
    checkpoint_path: str | None = None  # type: ignore[assignment]
    checkpoint_root: str = "pretrained/deltasplice"
    variant: str | None = DEFAULT_VARIANT
    convert_all: bool = False
    default_variant: str | None = DEFAULT_VARIANT

    def post(self):
        if self.convert_all:
            return
        variant = _resolve_variant(self.variant)
        if self.output_path is None:
            self.output_path = variant
        if self.checkpoint_path is None:
            self.checkpoint_path = str(Path(self.checkpoint_root) / VARIANT_DIRECTORIES[variant])
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    if config.convert_all:
        convert_all_checkpoints(config)
    else:
        convert_checkpoint(config)
