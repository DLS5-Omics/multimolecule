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
from typing import Any

import torch

import multimolecule.models.carp  # noqa: F401
from multimolecule.models.carp.configuration_carp import CarpConfig as Config
from multimolecule.models.carp.modeling_carp import CarpForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import (
    load_checkpoint,
    save_checkpoint,
    should_derive_output_path,
)
from multimolecule.tokenisers.protein.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

CARP_URL = "https://zenodo.org/record/6564798/files/"
VARIANTS = {
    "carp_600k": f"{CARP_URL}carp_600k.pt?download=1",
    "carp_38m": f"{CARP_URL}carp_38M.pt?download=1",
    "carp_76m": f"{CARP_URL}carp_76M.pt?download=1",
    "carp_640m": f"{CARP_URL}carp_640M.pt?download=1",
}
VARIANT_OUTPUT_NAMES = {
    "carp_600k": "carp-600k",
    "carp_38m": "carp-38m",
    "carp_76m": "carp-76m",
    "carp_640m": "carp-640m",
}

ORIGINAL_VOCAB_LIST: list[str] = list("ACDEFGHIKLMNPQRSTVWYBZXJOU") + [
    "<eos>",  # upstream STOP="*"
    "<pad>",  # upstream PAD="-"
    "<mask>",  # upstream MASK="#"
    "<cls>",  # upstream START="@"
]
ORIGINAL_RAW_VOCAB_LIST: list[str] = list("ACDEFGHIKLMNPQRSTVWYBZXJOU") + ["*", "-", "#", "@"]


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting CARP checkpoint at {convert_config.checkpoint_path}")
    vocab_list = list(get_alphabet().vocabulary)

    model_data = _load_model_data(convert_config.checkpoint_path)
    config = _build_config(model_data, vocab_list)
    state_dict = _convert_checkpoint(config, model_data["model_state_dict"], vocab_list, ORIGINAL_VOCAB_LIST)

    model = Model(config)
    load_checkpoint(model, state_dict)
    tokenizer_config = get_tokenizer_config()
    tokenizer_config.update({"bos_token": None, "eos_token": None, "sep_token": None})
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _build_config(model_data: dict[str, Any], vocab_list: list[str]) -> Config:
    hidden_size = int(model_data["d_model"])
    slim = bool(model_data["slim"])
    config = Config(
        vocab_size=len(vocab_list),
        embedding_size=int(model_data["d_embed"]),
        hidden_size=hidden_size,
        intermediate_size=hidden_size // 2 if slim else hidden_size,
        num_hidden_layers=int(model_data["n_layers"]),
        kernel_size=int(model_data["kernel_size"]),
        max_dilation=int(model_data["r"]),
        hidden_act=str(model_data["activation"]).lower(),
        slim=slim,
    )
    config.architectures = ["CarpForPreTraining"]
    return config


def _load_model_data(checkpoint_path: str) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return torch.load(path, map_location="cpu")

    variant = _normalize_variant(checkpoint_path)
    try:
        url = VARIANTS[variant]
    except KeyError as error:
        raise FileNotFoundError(
            f"CARP checkpoint {checkpoint_path!r} is not a file and is not one of {sorted(VARIANTS)}."
        ) from error

    return torch.hub.load_state_dict_from_url(
        url,
        model_dir=str(_scratch_dir()),
        progress=True,
        map_location="cpu",
    )


def _convert_checkpoint(
    config: Config,
    original_state_dict: dict[str, torch.Tensor],
    vocab_list: list[str],
    original_vocab_list: list[str],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    word_embeddings = None
    decoder_weight = None
    decoder_bias = None

    for key, value in original_state_dict.items():
        if key == "embedder.embedder.weight":
            word_embeddings = value
            continue
        if key == "embedder.up_embedder.conv.weight":
            state_dict["model.embeddings.projection.weight"] = value
            continue
        if key == "embedder.up_embedder.conv.bias":
            state_dict["model.embeddings.projection.bias"] = value
            continue
        if key == "last_norm.weight":
            state_dict["lm_head.layer_norm.weight"] = value
            continue
        if key == "last_norm.bias":
            state_dict["lm_head.layer_norm.bias"] = value
            continue
        if key == "decoder.conv.weight":
            decoder_weight = value.squeeze(-1).contiguous()
            continue
        if key == "decoder.conv.bias":
            decoder_bias = value
            continue
        if key.startswith("embedder.layers."):
            state_dict[_convert_layer_key(key)] = value
            continue
        raise KeyError(f"Unhandled CARP checkpoint key: {key}")

    if word_embeddings is None or decoder_weight is None or decoder_bias is None:
        raise KeyError("CARP checkpoint is missing word embeddings or decoder parameters.")

    word_embeddings, decoder_weight, decoder_bias = convert_word_embeddings(
        word_embeddings,
        decoder_weight,
        decoder_bias,
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embeddings
    state_dict["lm_head.decoder.weight"] = decoder_weight
    _copy_raw_token_rows(word_embeddings, decoder_weight, decoder_bias, original_state_dict, vocab_list)
    state_dict["lm_head.decoder.bias"] = decoder_bias
    return state_dict


def _copy_raw_token_rows(
    word_embeddings: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    original_state_dict: dict[str, torch.Tensor],
    vocab_list: list[str],
) -> None:
    original_word_embeddings = original_state_dict["embedder.embedder.weight"]
    original_decoder_weight = original_state_dict["decoder.conv.weight"].squeeze(-1)
    original_decoder_bias = original_state_dict["decoder.conv.bias"]
    for token in ("*", "-"):
        old_index = ORIGINAL_RAW_VOCAB_LIST.index(token)
        new_index = vocab_list.index(token)
        word_embeddings[new_index] = original_word_embeddings[old_index]
        decoder_weight[new_index] = original_decoder_weight[old_index]
        decoder_bias[new_index] = original_decoder_bias[old_index]


def _convert_layer_key(key: str) -> str:
    parts = key.split(".")
    if len(parts) < 5:
        raise KeyError(f"Malformed CARP layer key: {key}")
    layer_idx = parts[2]
    subkey = ".".join(parts[3:])
    base = f"model.encoder.layer.{layer_idx}"
    replacements = {
        "conv.weight": "convolution.weight",
        "conv.bias": "convolution.bias",
        "sequence1.0.weight": "layer_norm1.weight",
        "sequence1.0.bias": "layer_norm1.bias",
        "sequence1.2.conv.weight": "intermediate.weight",
        "sequence1.2.conv.bias": "intermediate.bias",
        "sequence1.3.weight": "layer_norm2.weight",
        "sequence1.3.bias": "layer_norm2.bias",
        "sequence2.0.weight": "layer_norm3.weight",
        "sequence2.0.bias": "layer_norm3.bias",
        "sequence2.2.conv.weight": "output.weight",
        "sequence2.2.conv.bias": "output.bias",
    }
    try:
        return f"{base}.{replacements[subkey]}"
    except KeyError as error:
        raise KeyError(f"Unhandled CARP layer key: {key}") from error


def _scratch_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "pretrained" / "carp"


def _normalize_variant(name: str) -> str:
    return Path(name).stem.lower().replace("-", "_")


class ConvertConfig(ConvertConfig_):
    checkpoint_path: str = ""
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "carp-600k"

    def post(self):
        if not self.checkpoint_path:
            raise ValueError("CARP converter requires --checkpoint_path.")
        derive_output_path = should_derive_output_path(self, Config.model_type)
        variant = _normalize_variant(self.checkpoint_path)
        if variant in VARIANTS and derive_output_path:
            self.output_path = VARIANT_OUTPUT_NAMES[variant]
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
