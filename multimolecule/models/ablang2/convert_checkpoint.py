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
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch

from multimolecule.models.ablang2 import AbLang2Config as Config
from multimolecule.models.ablang2 import AbLang2ForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.protein.utils import (
    convert_word_embeddings,
    get_alphabet,
    get_tokenizer_config,
)

torch.manual_seed(1016)

ORIGINAL_VOCAB_LIST: list[str] = [
    "<cls>",
    "M",
    "R",
    "H",
    "K",
    "D",
    "E",
    "S",
    "T",
    "N",
    "Q",
    "C",
    "G",
    "P",
    "A",
    "V",
    "I",
    "F",
    "Y",
    "W",
    "L",
    "<pad>",
    "<eos>",
    "<mask>",
    "X",
    "|",
]


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting AbLang2 checkpoint at {convert_config.checkpoint_path}")
    if convert_config.checkpoint_path is None:
        raise ValueError("AbLang2 converter requires --checkpoint_path.")
    checkpoint_path = _checkpoint_dir(convert_config.checkpoint_path)
    vocab_list = list(get_alphabet().vocabulary)

    with open(checkpoint_path / "hparams.json", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    config = _build_config(raw_config, vocab_list)
    try:
        state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu")
    state_dict = _convert_checkpoint(config, state_dict, vocab_list, ORIGINAL_VOCAB_LIST)

    model = Model(config)
    load_checkpoint(model, state_dict)
    model.tie_weights()
    tokenizer_config = get_tokenizer_config()
    tokenizer_config["sep_token"] = "|"
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _checkpoint_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_dir():
        return path
    if not path.is_file():
        raise FileNotFoundError(f"AbLang2 checkpoint path does not exist: {path}")
    if not tarfile.is_tarfile(path):
        raise ValueError(f"AbLang2 checkpoint path must be an extracted directory or tar archive: {path}")
    destination = Path(tempfile.mkdtemp(prefix="ablang2-checkpoint-")).resolve()
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Refusing to extract AbLang2 archive member outside destination: {member.name}")
            archive.extract(member, destination)
    return destination


def _build_config(raw_config: dict[str, Any], vocab_list: list[str]) -> Config:
    config = Config(
        vocab_size=len(vocab_list),
        hidden_size=int(raw_config.get("hidden_embed_size", 480)),
        num_hidden_layers=int(raw_config.get("n_encoder_blocks", 12)),
        num_attention_heads=int(raw_config.get("n_attn_heads", 20)),
        intermediate_size=4 * int(raw_config.get("hidden_embed_size", 480)),
        hidden_act=str(raw_config.get("a_fn", "swiglu")).lower(),
        hidden_dropout=float(raw_config.get("dropout", 0.0)),
        attention_dropout=float(raw_config.get("dropout", 0.0)),
        layer_norm_eps=float(raw_config.get("layer_norm_eps", 1e-12)),
        attention_bias=True,
        feedforward_bias=True,
    )
    config.architectures = ["AbLang2ForPreTraining"]
    return config


def _convert_checkpoint(
    config: Config,
    original_state_dict: dict[str, torch.Tensor],
    vocab_list: list[str],
    original_vocab_list: list[str],
) -> dict[str, torch.Tensor]:
    if not torch.equal(
        original_state_dict["AbRep.aa_embed_layer.weight"],
        original_state_dict["AbHead.weights"],
    ):
        raise ValueError("AbLang2 checkpoint does not tie AbHead.weights to AbRep.aa_embed_layer.weight.")

    state_dict: dict[str, torch.Tensor] = {}
    word_embeddings, decoder_bias = convert_word_embeddings(
        original_state_dict["AbRep.aa_embed_layer.weight"],
        original_state_dict["AbHead.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embeddings
    state_dict["lm_head.decoder.weight"] = word_embeddings
    state_dict["lm_head.bias"] = decoder_bias
    state_dict["lm_head.decoder.bias"] = decoder_bias

    for key, value in original_state_dict.items():
        if key in {
            "AbRep.aa_embed_layer.weight",
            "AbHead.weights",
            "AbHead.bias",
        }:
            continue
        if key == "AbRep.layer_norm_after_encoder_blocks.weight":
            state_dict["model.encoder.layer_norm.weight"] = value
            continue
        if key == "AbRep.layer_norm_after_encoder_blocks.bias":
            state_dict["model.encoder.layer_norm.bias"] = value
            continue
        if key == "AbHead.ff.0.weight":
            state_dict["lm_head.dense.weight"] = value
            continue
        if key == "AbHead.ff.0.bias":
            state_dict["lm_head.dense.bias"] = value
            continue
        if key == "AbHead.ff.2.weight":
            state_dict["lm_head.layer_norm.weight"] = value
            continue
        if key == "AbHead.ff.2.bias":
            state_dict["lm_head.layer_norm.bias"] = value
            continue
        if key.startswith("AbRep.encoder_blocks."):
            _convert_encoder_block_key(state_dict, key, value)
            continue
        raise KeyError(f"Unhandled checkpoint key: {key}")

    if word_embeddings.shape[0] != config.vocab_size:
        raise ValueError(f"Converted embedding has {word_embeddings.shape[0]} rows; expected {config.vocab_size}.")
    return state_dict


def _convert_encoder_block_key(state_dict: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    parts = key.split(".")
    if len(parts) < 4:
        raise KeyError(f"Unhandled encoder-block key: {key}")
    layer_idx = parts[2]
    sub = ".".join(parts[3:])
    base = f"model.encoder.layer.{layer_idx}"
    if sub == "multihead_attention.rotary_emb.freqs":
        return
    if sub == "multihead_attention.q_proj.weight":
        state_dict[f"{base}.attention.self.query.weight"] = value
    elif sub == "multihead_attention.q_proj.bias":
        state_dict[f"{base}.attention.self.query.bias"] = value
    elif sub == "multihead_attention.k_proj.weight":
        state_dict[f"{base}.attention.self.key.weight"] = value
    elif sub == "multihead_attention.k_proj.bias":
        state_dict[f"{base}.attention.self.key.bias"] = value
    elif sub == "multihead_attention.v_proj.weight":
        state_dict[f"{base}.attention.self.value.weight"] = value
    elif sub == "multihead_attention.v_proj.bias":
        state_dict[f"{base}.attention.self.value.bias"] = value
    elif sub == "multihead_attention.out_proj.weight":
        state_dict[f"{base}.attention.output.dense.weight"] = value
    elif sub == "multihead_attention.out_proj.bias":
        state_dict[f"{base}.attention.output.dense.bias"] = value
    elif sub == "intermediate_layer.0.weight":
        state_dict[f"{base}.intermediate.dense.weight"] = value
    elif sub == "intermediate_layer.0.bias":
        state_dict[f"{base}.intermediate.dense.bias"] = value
    elif sub == "intermediate_layer.2.weight":
        state_dict[f"{base}.output.dense.weight"] = value
    elif sub == "intermediate_layer.2.bias":
        state_dict[f"{base}.output.dense.bias"] = value
    elif sub == "pre_attn_layer_norm.weight":
        state_dict[f"{base}.attention.layer_norm.weight"] = value
    elif sub == "pre_attn_layer_norm.bias":
        state_dict[f"{base}.attention.layer_norm.bias"] = value
    elif sub == "final_layer_norm.weight":
        state_dict[f"{base}.layer_norm.weight"] = value
    elif sub == "final_layer_norm.bias":
        state_dict[f"{base}.layer_norm.bias"] = value
    else:
        raise KeyError(f"Unhandled encoder-block key: {key}")


class ConvertConfig(ConvertConfig_):
    checkpoint_path: str | None = None  # type: ignore[assignment]
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "ablang2"

    def post(self):
        if self.checkpoint_path is None:
            raise ValueError("AbLang2 converter requires --checkpoint_path.")
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
