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

import glob
import os
from pathlib import Path
from typing import Any

import chanfig
import torch

from multimolecule.models import EsmCConfig as Config
from multimolecule.models import EsmCForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.protein.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Biohub ESMC tokenizer order. The model embedding/head matrices are 64 rows,
# while only the first 33 are currently assigned concrete tokens.
ORIGINAL_VOCAB_LIST: list[str | None] = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
    *([None] * 31),
]


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting ESMC checkpoint at {convert_config.checkpoint_path}")
    vocab_list = list(get_alphabet().vocabulary)

    raw_config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config = _build_config(raw_config, vocab_list)
    state_dict = _load_state_dict(convert_config.checkpoint_path)
    state_dict = _convert_checkpoint(config, state_dict, vocab_list, ORIGINAL_VOCAB_LIST)

    model = Model(config)
    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=get_tokenizer_config())
    print(f"Checkpoint saved to {convert_config.output_path}")


def _build_config(raw_config: dict[str, Any], vocab_list: list[str]) -> Config:
    config = Config(
        vocab_size=len(vocab_list),
        hidden_size=int(raw_config.get("hidden_size", raw_config.get("d_model", 960))),
        num_hidden_layers=int(raw_config.get("num_hidden_layers", raw_config.get("n_layers", 30))),
        num_attention_heads=int(raw_config.get("num_attention_heads", raw_config.get("n_heads", 15))),
        intermediate_size=raw_config.get("intermediate_size"),
        hidden_dropout=float(raw_config.get("hidden_dropout", 0.0)),
        attention_dropout=float(raw_config.get("attention_dropout", 0.0)),
        max_position_embeddings=int(raw_config.get("max_position_embeddings", 2048)),
        initializer_range=float(raw_config.get("initializer_range", 0.02)),
        layer_norm_eps=float(raw_config.get("layer_norm_eps", 1e-5)),
        attention_bias=bool(raw_config.get("attention_bias", False)),
        feedforward_bias=bool(raw_config.get("feedforward_bias", False)),
        attention_layer_norm_bias=bool(raw_config.get("attention_layer_norm_bias", True)),
        qk_layer_norm=bool(raw_config.get("qk_layer_norm", True)),
        qk_layer_norm_bias=bool(raw_config.get("qk_layer_norm_bias", False)),
        final_layer_norm_bias=bool(raw_config.get("final_layer_norm_bias", False)),
        residue_scaling_factor=raw_config.get("residue_scaling_factor"),
    )
    config.architectures = ["EsmCForPreTraining"]
    return config


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    safetensor_paths = sorted(glob.glob(os.path.join(checkpoint_path, "model*.safetensors")))
    if safetensor_paths:
        from safetensors.torch import load_file

        for safetensors_path in safetensor_paths:
            state_dict.update(load_file(safetensors_path))
        return state_dict
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model weights found under {checkpoint_path}")


def _convert_checkpoint(
    config: Config,
    original_state_dict: dict[str, torch.Tensor],
    vocab_list: list[str],
    original_vocab_list: list[str | None],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in original_state_dict.items():
        key = key.removeprefix("esmc.")
        if key.endswith("._extra_state"):
            continue
        if key in ("embed.weight", "embed_tokens.weight"):
            state_dict["model.embeddings.word_embeddings.weight"] = value
            continue
        if key in ("transformer.norm.weight", "norm.weight"):
            state_dict["model.encoder.layer_norm.weight"] = value
            continue
        if key.startswith(("sequence_head.", "lm_head.")):
            _convert_sequence_head_key(state_dict, key, value)
            continue
        if key.startswith(("transformer.blocks.", "layers.")):
            _convert_block_key(config, state_dict, key, value)
            continue
        raise KeyError(f"Unhandled checkpoint key: {key}")

    word_embed = state_dict["model.embeddings.word_embeddings.weight"]
    decoder_weight = state_dict["lm_head.decoder.weight"]
    decoder_bias = state_dict["lm_head.decoder.bias"]
    word_embed, decoder_weight, decoder_bias = convert_word_embeddings(
        word_embed,
        decoder_weight,
        decoder_bias,
        old_vocab=original_vocab_list,  # type: ignore[arg-type]
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embed
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias

    if word_embed.shape[0] != config.vocab_size:
        raise ValueError(f"Converted embedding has {word_embed.shape[0]} rows; expected {config.vocab_size}.")
    return state_dict


def _convert_sequence_head_key(state_dict: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    key = key.removeprefix("sequence_head.").removeprefix("lm_head.")
    names = {
        "0.weight": "transform.dense.weight",
        "0.bias": "transform.dense.bias",
        "dense.weight": "transform.dense.weight",
        "dense.bias": "transform.dense.bias",
        "2.weight": "transform.layer_norm.weight",
        "2.bias": "transform.layer_norm.bias",
        "layer_norm.weight": "transform.layer_norm.weight",
        "layer_norm.bias": "transform.layer_norm.bias",
        "3.weight": "decoder.weight",
        "3.bias": "decoder.bias",
        "decoder.weight": "decoder.weight",
        "decoder.bias": "decoder.bias",
    }
    state_dict[f"lm_head.{names[key]}"] = value


def _convert_block_key(
    config: Config,
    state_dict: dict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> None:
    prefix = "transformer.blocks." if key.startswith("transformer.blocks.") else "layers."
    index, sub = key.removeprefix(prefix).split(".", 1)
    base = f"model.encoder.layer.{index}"
    if sub in ("attn.layernorm_qkv.weight", "attn.layernorm_qkv.1.weight"):
        if value.shape[0] != 3 * config.hidden_size:
            raise ValueError(
                f"Unexpected QKV weight shape {tuple(value.shape)}; expected first dim {3 * config.hidden_size}."
            )
        query, key_, value_ = value.split(config.hidden_size, dim=0)
        state_dict[f"{base}.attention.self.query.weight"] = query.clone()
        state_dict[f"{base}.attention.self.key.weight"] = key_.clone()
        state_dict[f"{base}.attention.self.value.weight"] = value_.clone()
    elif sub in ("ffn.fc1_weight", "ffn.1.weight"):
        if value.shape[0] != 2 * config.intermediate_size:
            raise ValueError(
                f"Unexpected fc1_weight shape {tuple(value.shape)}; expected first dim "
                f"{2 * config.intermediate_size}."
            )
        gate, up = value.split(config.intermediate_size, dim=0)
        state_dict[f"{base}.intermediate.gate_proj.weight"] = gate.clone()
        state_dict[f"{base}.intermediate.up_proj.weight"] = up.clone()
    else:
        names = {
            "attn.layernorm_qkv.layer_norm_weight": "attention.layer_norm.weight",
            "attn.layernorm_qkv.layer_norm_bias": "attention.layer_norm.bias",
            "attn.layernorm_qkv.0.weight": "attention.layer_norm.weight",
            "attn.layernorm_qkv.0.bias": "attention.layer_norm.bias",
            "input_layernorm.weight": "attention.layer_norm.weight",
            "input_layernorm.bias": "attention.layer_norm.bias",
            "attn.q_ln.weight": "attention.self.query_layer_norm.weight",
            "attn.k_ln.weight": "attention.self.key_layer_norm.weight",
            "self_attn.q_norm.weight": "attention.self.query_layer_norm.weight",
            "self_attn.k_norm.weight": "attention.self.key_layer_norm.weight",
            "self_attn.q_proj.weight": "attention.self.query.weight",
            "self_attn.k_proj.weight": "attention.self.key.weight",
            "self_attn.v_proj.weight": "attention.self.value.weight",
            "attn.out_proj.weight": "attention.output.dense.weight",
            "self_attn.o_proj.weight": "attention.output.dense.weight",
            "ffn.layer_norm_weight": "layer_norm.weight",
            "ffn.layer_norm_bias": "layer_norm.bias",
            "ffn.0.weight": "layer_norm.weight",
            "ffn.0.bias": "layer_norm.bias",
            "post_attention_layernorm.weight": "layer_norm.weight",
            "post_attention_layernorm.bias": "layer_norm.bias",
            "mlp.gate_proj.weight": "intermediate.gate_proj.weight",
            "mlp.up_proj.weight": "intermediate.up_proj.weight",
            "ffn.fc2_weight": "output.dense.weight",
            "ffn.3.weight": "output.dense.weight",
            "mlp.down_proj.weight": "output.dense.weight",
        }
        state_dict[f"{base}.{names[sub]}"] = value


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "esmc-300m"

    def post(self):
        if self.output_path != Config.model_type:
            return super().post()
        checkpoint_path = Path(self.checkpoint_path).name.lower()
        if "6b" in checkpoint_path:
            suffix = "6b"
        elif "600m" in checkpoint_path:
            suffix = "600m"
        elif "300m" in checkpoint_path:
            suffix = "300m"
        else:
            raise ValueError(f"Cannot infer ESMC variant size from {self.checkpoint_path}")
        if not self.output_path.endswith(suffix):
            self.output_path = f"{Config.model_type}-{suffix}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
