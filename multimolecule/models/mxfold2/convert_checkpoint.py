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
from pathlib import Path

import torch

from multimolecule.models import Mxfold2Config as Config
from multimolecule.models import Mxfold2Model as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import get_alphabet, get_tokenizer_config


def convert_checkpoint(convert_config):
    conf_path, param_path = _resolve_paths(convert_config.checkpoint_path)
    parsed = _parse_original_config(conf_path)

    config = Config(
        folding_model=parsed.get("model", "MixC"),
        max_helix_length=parsed.get("max_helix_length", 30),
        embed_size=parsed.get("embed_size", 64),
        num_filters=parsed.get("num_filters", [64] * 8),
        filter_size=parsed.get("filter_size", [5, 3, 5, 3, 5, 3, 5, 3]),
        pool_size=parsed.get("pool_size", [1]),
        dilation=parsed.get("dilation", 0),
        num_lstm_layers=parsed.get("num_lstm_layers", 2),
        num_lstm_units=parsed.get("num_lstm_units", 32),
        num_transformer_layers=parsed.get("num_transformer_layers", 0),
        num_transformer_hidden_units=parsed.get("num_transformer_hidden_units", 2048),
        num_transformer_att=parsed.get("num_transformer_att", 8),
        num_hidden_units=parsed.get("num_hidden_units", [32]),
        num_paired_filters=parsed.get("num_paired_filters", [64] * 8),
        paired_filter_size=parsed.get("paired_filter_size", [5, 3, 5, 3, 5, 3, 5, 3]),
        dropout_rate=parsed.get("dropout_rate", 0.5),
        fc_dropout_rate=parsed.get("fc_dropout_rate", 0.5),
        num_att=parsed.get("num_att", 8),
        pair_join=parsed.get("pair_join", "cat"),
        no_split_lr=parsed.get("no_split_lr", False),
    )
    config.architectures = ["Mxfold2Model"]

    model = Model(config)

    state_dict = torch.load(param_path, map_location=torch.device("cpu"))
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    state_dict = {f"model.{key}": value.float() for key, value in state_dict.items()}

    load_checkpoint(model, state_dict)

    tokenizer_config = get_tokenizer_config()
    tokenizer_config["alphabet"] = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)


def _resolve_paths(checkpoint_path: str) -> tuple[Path, Path]:
    path = Path(checkpoint_path)
    if path.is_dir():
        conf_path = path / "TrainSetAB.conf"
        param_path = path / "TrainSetAB.pth"
        return conf_path, param_path
    if path.suffix == ".conf":
        return path, path.with_suffix(".pth")
    if path.suffix == ".pth":
        return path.with_suffix(".conf"), path
    raise ValueError(f"Unsupported MXfold2 checkpoint path: {checkpoint_path}")


def _parse_original_config(conf_path: Path) -> dict[str, object]:
    lines = [line.strip() for line in conf_path.read_text().splitlines() if line.strip()]
    parsed: dict[str, object] = {}
    index = 0
    while index < len(lines):
        key = lines[index]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected line in MXfold2 config {conf_path}: {key}")
        key = key[2:].replace("-", "_")
        if index + 1 < len(lines) and not lines[index + 1].startswith("--"):
            value = lines[index + 1]
            index += 2
        else:
            value = True
            index += 1
        if key in parsed:
            if not isinstance(parsed[key], list):
                parsed[key] = [parsed[key]]
            parsed[key].append(value)
        else:
            parsed[key] = value

    int_keys = {
        "max_helix_length",
        "embed_size",
        "dilation",
        "num_lstm_layers",
        "num_lstm_units",
        "num_transformer_layers",
        "num_transformer_hidden_units",
        "num_transformer_att",
        "num_att",
    }
    float_keys = {"dropout_rate", "fc_dropout_rate"}
    list_int_keys = {
        "num_filters",
        "filter_size",
        "pool_size",
        "num_hidden_units",
        "num_paired_filters",
        "paired_filter_size",
    }
    for key in int_keys:
        if key in parsed:
            parsed[key] = int(parsed[key])  # type: ignore[arg-type]
    for key in float_keys:
        if key in parsed:
            parsed[key] = float(parsed[key])  # type: ignore[arg-type]
    for key in list_int_keys:
        if key in parsed:
            values = parsed[key] if isinstance(parsed[key], list) else [parsed[key]]
            parsed[key] = [int(value) for value in values]
    if "no_split_lr" in parsed:
        parsed["no_split_lr"] = bool(parsed["no_split_lr"])
    return parsed


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
