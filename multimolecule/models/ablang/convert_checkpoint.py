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
from pathlib import Path
from typing import Any

import torch

from multimolecule.models.ablang.configuration_ablang import AbLangConfig as Config
from multimolecule.models.ablang.modeling_ablang import AbLangForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import append_output_suffix, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.protein.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

ORIGINAL_VOCAB_MAP: dict[str, str] = {
    "<": "<cls>",
    "-": "<pad>",
    ">": "<eos>",
    "*": "<mask>",
}


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting AbLang checkpoint at {convert_config.checkpoint_path}")
    checkpoint_path = Path(convert_config.checkpoint_path)
    raw_config = _load_json(checkpoint_path / "hparams.json")
    original_vocab = _load_original_vocab(checkpoint_path / "vocab.json")
    vocab_list = list(get_alphabet().vocabulary)

    config = _build_config(raw_config, vocab_list)
    state_dict = _load_state_dict(checkpoint_path / "amodel.pt")
    state_dict = _convert_checkpoint(config, state_dict, vocab_list, original_vocab)

    model = Model(config)
    load_checkpoint(model, state_dict)
    tokenizer_config = get_tokenizer_config()
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 1
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_original_vocab(path: Path) -> list[str]:
    raw_vocab = _load_json(path)
    vocab: list[str | None] = [None] * len(raw_vocab)
    for token, index in raw_vocab.items():
        vocab[int(index)] = ORIGINAL_VOCAB_MAP.get(token, token)
    if any(token is None for token in vocab):
        raise ValueError(f"{path} does not define a contiguous vocabulary")
    return [str(token) for token in vocab]


def _build_config(raw_config: dict[str, Any], vocab_list: list[str]) -> Config:
    config = Config(
        vocab_size=len(vocab_list),
        hidden_size=int(raw_config.get("hidden_size", 768)),
        num_hidden_layers=int(raw_config.get("num_hidden_layers", 12)),
        num_attention_heads=int(raw_config.get("num_attention_heads", 12)),
        intermediate_size=int(raw_config.get("intermediate_size", 3072)),
        hidden_act=str(raw_config.get("hidden_act", "gelu")).lower(),
        hidden_dropout=float(raw_config.get("hidden_dropout_prob", 0.1)),
        attention_dropout=float(raw_config.get("attention_probs_dropout_prob", 0.1)),
        max_position_embeddings=int(raw_config.get("max_position_embeddings", 160)),
        initializer_range=float(raw_config.get("initializer_range", 0.02)),
        layer_norm_eps=float(raw_config.get("layer_norm_eps", 1e-12)),
        chain=str(raw_config["chain"]) if raw_config.get("chain") is not None else None,
    )
    config.architectures = ["AbLangForPreTraining"]
    return config


def _convert_checkpoint(
    config: Config,
    original_state_dict: dict[str, torch.Tensor],
    vocab_list: list[str],
    original_vocab_list: list[str],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    decoder_bias: torch.Tensor | None = None
    for key, value in original_state_dict.items():
        if key == "AbRep.AbEmbeddings.AAEmbeddings.weight":
            state_dict["model.embeddings.word_embeddings.weight"] = value
        elif key == "AbRep.AbEmbeddings.PositionEmbeddings.weight":
            state_dict["model.embeddings.position_embeddings.weight"] = value
        elif key.startswith("AbRep.AbEmbeddings.LayerNorm."):
            state_dict[key.replace("AbRep.AbEmbeddings.LayerNorm", "model.embeddings.layer_norm")] = value
        elif key.startswith("AbRep.EncoderBlocks.Layers."):
            new_key = _convert_encoder_key(key)
            state_dict[new_key] = value
        elif key == "AbHead.bias":
            decoder_bias = value
        elif key == "AbHead.decoder.bias":
            if decoder_bias is None:
                decoder_bias = value
        elif key.startswith("AbHead.dense."):
            state_dict[key.replace("AbHead.dense", "lm_head.transform.dense")] = value
        elif key.startswith("AbHead.layer_norm."):
            state_dict[key.replace("AbHead.layer_norm", "lm_head.transform.layer_norm")] = value
        elif key == "AbHead.decoder.weight":
            state_dict["lm_head.decoder.weight"] = value
        else:
            raise KeyError(f"Unhandled AbLang checkpoint key: {key}")

    if decoder_bias is None:
        raise KeyError("AbLang checkpoint is missing AbHead.bias")

    word_embeddings = state_dict["model.embeddings.word_embeddings.weight"]
    decoder_weight = state_dict["lm_head.decoder.weight"]
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
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias

    if word_embeddings.shape[0] != config.vocab_size:
        raise ValueError(f"Converted embedding has {word_embeddings.shape[0]} rows; expected {config.vocab_size}.")
    return state_dict


def _convert_encoder_key(key: str) -> str:
    parts = key.split(".")
    layer_idx = parts[3]
    subkey = ".".join(parts[4:])
    base = f"model.encoder.layer.{layer_idx}"
    replacements = {
        "MultiHeadAttention.Attention.q_proj.weight": "attention.self.query.weight",
        "MultiHeadAttention.Attention.q_proj.bias": "attention.self.query.bias",
        "MultiHeadAttention.Attention.k_proj.weight": "attention.self.key.weight",
        "MultiHeadAttention.Attention.k_proj.bias": "attention.self.key.bias",
        "MultiHeadAttention.Attention.v_proj.weight": "attention.self.value.weight",
        "MultiHeadAttention.Attention.v_proj.bias": "attention.self.value.bias",
        "MultiHeadAttention.Attention.out_proj.weight": "attention.output.dense.weight",
        "MultiHeadAttention.Attention.out_proj.bias": "attention.output.dense.bias",
        "MHALayerNorm.weight": "attention.layer_norm.weight",
        "MHALayerNorm.bias": "attention.layer_norm.bias",
        "IntermediateLayer.expand_dense.weight": "intermediate.dense.weight",
        "IntermediateLayer.expand_dense.bias": "intermediate.dense.bias",
        "IntermediateLayer.dense_dense.weight": "output.dense.weight",
        "IntermediateLayer.dense_dense.bias": "output.dense.bias",
        "IntermediateLayer.LayerNorm.weight": "output.layer_norm.weight",
        "IntermediateLayer.LayerNorm.bias": "output.layer_norm.bias",
    }
    try:
        return f"{base}.{replacements[subkey]}"
    except KeyError as error:
        raise KeyError(f"Unhandled AbLang encoder key: {key}") from error


class ConvertConfig(ConvertConfig_):
    checkpoint_path: str = ""
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "ablang-heavy"

    def post(self):
        checkpoint_parts = {part.lower() for part in Path(self.checkpoint_path).parts}
        if "heavy" in checkpoint_parts:
            suffix = "heavy"
        elif "light" in checkpoint_parts:
            suffix = "light"
        else:
            raise ValueError(f"Cannot infer AbLang chain variant from {self.checkpoint_path}")
        append_output_suffix(self, suffix)
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
