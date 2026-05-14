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
from typing import Dict, List

import chanfig
import torch

from multimolecule.models import AMPLIFYConfig as Config
from multimolecule.models import AMPLIFYForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.protein.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# AMPLIFY upstream vocabulary order from chandar-lab/AMPLIFY tokenizer.json.
ORIGINAL_VOCAB_LIST: List[str] = [
    "<pad>",
    "<unk>",
    "<mask>",
    "<cls>",  # AMPLIFY's ``<bos>`` plays the same role as MultiMolecule's ``<cls>``.
    "<eos>",
    "|",  # chain separator used in AMPLIFY's multi-chain inputs (e.g. antibody H | L).
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
    "B",
]


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting AMPLIFY checkpoint at {convert_config.checkpoint_path}")
    vocab_list = list(get_alphabet().vocabulary)

    raw_config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config = _build_config(raw_config, vocab_list)
    state_dict = _load_state_dict(convert_config.checkpoint_path)
    state_dict = _convert_checkpoint(config, state_dict, vocab_list, ORIGINAL_VOCAB_LIST)

    model = Model(config)
    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=get_tokenizer_config())
    print(f"Checkpoint saved to {convert_config.output_path}")


def _build_config(raw_config: Dict, vocab_list: List[str]) -> Config:
    config = Config(
        vocab_size=len(vocab_list),
        hidden_size=int(raw_config.get("hidden_size", 640)),
        num_hidden_layers=int(raw_config.get("num_hidden_layers", 24)),
        num_attention_heads=int(raw_config.get("num_attention_heads", 10)),
        intermediate_size=int(raw_config.get("intermediate_size", 2560)),
        hidden_act=str(raw_config.get("hidden_act", "swiglu")).lower(),
        hidden_dropout=float(raw_config.get("dropout_prob", 0.0)),
        attention_dropout=float(raw_config.get("dropout_prob", 0.0)),
        max_position_embeddings=int(raw_config.get("max_length", 2048)),
        initializer_range=float(raw_config.get("decoder_init_range", 0.02)),
        layer_norm_eps=float(raw_config.get("norm_eps", 1e-5)),
        rms_norm=bool(raw_config.get("rms_norm", True)),
        layer_norm_after_embedding=bool(raw_config.get("layer_norm_after_embedding", False)),
        layer_norm_before_last_layer=bool(raw_config.get("layer_norm_before_last_layer", True)),
        attention_bias=bool(raw_config.get("att_bias", False)),
        feedforward_bias=bool(raw_config.get("ffn_bias", False)),
    )
    config.architectures = ["AMPLIFYForPreTraining"]
    return config


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path)
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found under {checkpoint_path}")


def _convert_checkpoint(
    config: Config,
    original_state_dict: Dict[str, torch.Tensor],
    vocab_list: List[str],
    original_vocab_list: List[str],
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    intermediate_size = _swiglu_intermediate_size(config.intermediate_size)
    for key, value in original_state_dict.items():
        # Word embeddings -> model.embeddings.word_embeddings
        if key == "encoder.weight":
            state_dict["model.embeddings.word_embeddings.weight"] = value
            continue
        # Optional post-embedding norm
        if key.startswith("layer_norm_1"):
            new_key = key.replace("layer_norm_1", "model.embeddings.layer_norm")
            state_dict[new_key] = value
            continue
        # Final encoder norm before the LM head
        if key.startswith("layer_norm_2"):
            new_key = key.replace("layer_norm_2", "model.encoder.layer_norm")
            state_dict[new_key] = value
            continue
        # LM head decoder
        if key.startswith("decoder."):
            new_key = key.replace("decoder", "lm_head.decoder")
            state_dict[new_key] = value
            continue
        # Encoder blocks
        if key.startswith("transformer_encoder."):
            parts = key.split(".")
            layer_idx, sub = parts[1], ".".join(parts[2:])
            base = f"model.encoder.layer.{layer_idx}"
            if sub == "q.weight":
                state_dict[f"{base}.attention.q_proj.weight"] = value
            elif sub == "k.weight":
                state_dict[f"{base}.attention.k_proj.weight"] = value
            elif sub == "v.weight":
                state_dict[f"{base}.attention.v_proj.weight"] = value
            elif sub == "wo.weight":
                state_dict[f"{base}.attention.out_proj.weight"] = value
            elif sub == "q.bias":
                state_dict[f"{base}.attention.q_proj.bias"] = value
            elif sub == "k.bias":
                state_dict[f"{base}.attention.k_proj.bias"] = value
            elif sub == "v.bias":
                state_dict[f"{base}.attention.v_proj.bias"] = value
            elif sub == "wo.bias":
                state_dict[f"{base}.attention.out_proj.bias"] = value
            elif sub == "attention_norm.weight":
                state_dict[f"{base}.attention_norm.weight"] = value
            elif sub == "ffn_norm.weight":
                state_dict[f"{base}.ffn_norm.weight"] = value
            elif sub == "ffn.w12.weight":
                # xformers.SwiGLU stacks ``gate`` and ``up`` projections in a
                # single weight: ``[gate; up]``. Split them along dim 0.
                if value.shape[0] != 2 * intermediate_size:
                    raise ValueError(
                        f"Unexpected w12.weight shape {tuple(value.shape)}; expected first dim "
                        f"{2 * intermediate_size}."
                    )
                gate, up = value.chunk(2, dim=0)
                state_dict[f"{base}.mlp.gate_proj.weight"] = gate.contiguous()
                state_dict[f"{base}.mlp.up_proj.weight"] = up.contiguous()
            elif sub == "ffn.w12.bias":
                gate_b, up_b = value.chunk(2, dim=0)
                state_dict[f"{base}.mlp.gate_proj.bias"] = gate_b.contiguous()
                state_dict[f"{base}.mlp.up_proj.bias"] = up_b.contiguous()
            elif sub == "ffn.w3.weight":
                state_dict[f"{base}.mlp.down_proj.weight"] = value
            elif sub == "ffn.w3.bias":
                state_dict[f"{base}.mlp.down_proj.bias"] = value
            else:
                raise KeyError(f"Unhandled encoder-block key: {key}")
            continue
        raise KeyError(f"Unhandled checkpoint key: {key}")

    # Reorder the vocabulary-dependent rows so the converted model speaks the
    # MultiMolecule protein vocabulary.
    word_embed = state_dict["model.embeddings.word_embeddings.weight"]
    decoder_weight = state_dict["lm_head.decoder.weight"]
    decoder_bias = state_dict["lm_head.decoder.bias"]
    word_embed, decoder_weight, decoder_bias = convert_word_embeddings(
        word_embed,
        decoder_weight,
        decoder_bias,
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embed
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias

    if word_embed.shape[0] != config.vocab_size:
        raise ValueError(
            f"Converted embedding has {word_embed.shape[0]} rows; expected {config.vocab_size}."
        )
    return state_dict


def _swiglu_intermediate_size(intermediate_size: int, multiple_of: int = 8) -> int:
    reduced = int(2 * intermediate_size / 3)
    return multiple_of * ((reduced + multiple_of - 1) // multiple_of)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "amplify-120m"

    def post(self):
        checkpoint_path = Path(self.checkpoint_path).name.lower()
        if "350m" in checkpoint_path:
            suffix = "350m"
        elif "120m" in checkpoint_path:
            suffix = "120m"
        else:
            raise ValueError(f"Cannot infer AMPLIFY variant size from {self.checkpoint_path}")
        if "base" in checkpoint_path:
            suffix += "-base"
        if not self.output_path.endswith(suffix):
            self.output_path = f"{Config.model_type}-{suffix}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
