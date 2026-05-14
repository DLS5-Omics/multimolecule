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
from typing import List

import chanfig
import torch
from safetensors.torch import load_file

from multimolecule.models import GenerannoConfig as Config
from multimolecule.models import GenerannoForMaskedLM as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Order of the original GENERanno vocab (vocab.txt entries 0..42).
# These IUPAC ambiguity codes (`<K>`, `<M>`, ..., `<Y>`) are wrapped in angle brackets in the upstream
# vocab but represent the standard single-letter ambiguity codes; we map them back to plain letters.
ORIGINAL_VOCAB_LIST: List[str] = [
    "<oov>",  # 0  (unused at runtime; treat as <unk>)
    "<cls>",  # 1  (upstream <s>)
    "<eos>",  # 2  (upstream </s>)
    "<pad>",  # 3
    "<mask>",  # 4
    "<bog>",  # 5
    "<eog>",  # 6
    "<bok>",  # 7
    "<eok>",  # 8
    "<+>",  # 9
    "<->",  # 10
    "<mam>",  # 11
    "<vrt>",  # 12
    "<inv>",  # 13
    "<pln>",  # 14
    "<fng>",  # 15
    "<prt>",  # 16
    "<arc>",  # 17
    "<bct>",  # 18
    "<mit>",  # 19
    "<plt>",  # 20
    "<plm>",  # 21
    "<vir>",  # 22
    "<cds>",  # 23
    "<pseudo>",  # 24
    "<tRNA>",  # 25
    "<rRNA>",  # 26
    "<ncRNA>",  # 27
    "<sp0>",  # 28
    "<sp1>",  # 29
    "<sp2>",  # 30
    "<sp3>",  # 31
    "A",  # 32
    "C",  # 33
    "G",  # 34
    "K",  # 35  (upstream <K>)
    "M",  # 36  (upstream <M>)
    "N",  # 37
    "R",  # 38  (upstream <R>)
    "S",  # 39  (upstream <S>)
    "T",  # 40
    "W",  # 41  (upstream <W>)
    "Y",  # 42  (upstream <Y>)
]


def convert_checkpoint(convert_config: ConvertConfig) -> None:
    print(f"Converting GENERanno checkpoint at {convert_config.checkpoint_path}")
    original_config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config = _build_mm_config(original_config)

    alphabet = get_alphabet("standard")
    new_vocab = list(alphabet.vocabulary)
    config.vocab_size = len(new_vocab)
    config.architectures = ["GenerannoForMaskedLM"]

    model = Model(config)

    safetensors_path = os.path.join(convert_config.checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        ckpt = load_file(safetensors_path)
    else:
        ckpt = torch.load(
            os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
        )

    state_dict = _convert_checkpoint(config, ckpt, ORIGINAL_VOCAB_LIST, new_vocab)
    load_checkpoint(model, state_dict)

    tokenizer_config = get_tokenizer_config()
    tokenizer_config["alphabet"] = "standard"
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _build_mm_config(original_config: chanfig.FlatDict) -> Config:
    fields = {
        "vocab_size": original_config.get("vocab_size", 64),
        "hidden_size": original_config.get("hidden_size", 1280),
        "num_hidden_layers": original_config.get("num_hidden_layers", 28),
        "num_attention_heads": original_config.get("num_attention_heads", 16),
        "num_key_value_heads": original_config.get("num_key_value_heads", 4),
        "intermediate_size": original_config.get("intermediate_size", 3520),
        "hidden_act": original_config.get("hidden_act", "silu"),
        "attention_dropout": original_config.get("attention_dropout", 0.0),
        "max_position_embeddings": original_config.get("max_position_embeddings", 8192),
        "initializer_range": original_config.get("initializer_range", 0.02),
        "rms_norm_eps": original_config.get("rms_norm_eps", 1e-5),
        "attention_bias": original_config.get("attention_bias", False),
        "mlp_bias": original_config.get("mlp_bias", False),
        "rope_theta": original_config.get("rope_theta", 500000.0),
    }
    config = Config(**fields)
    del config._name_or_path
    return config


def _convert_checkpoint(config, original_state_dict, original_vocab, new_vocab):
    state_dict = {}
    for key, value in original_state_dict.items():
        new_key = convert_original_state_dict_key(key)
        if new_key is None:
            continue
        state_dict[new_key] = value

    embed_key = "model.embeddings.word_embeddings.weight"
    decoder_key = "lm_head.decoder.weight"

    # The upstream embedding matrix is padded beyond the 43 named tokens (e.g. to vocab_size=64).
    # Rows for upstream-only or unused tokens are skipped during vocabulary remapping by setting their
    # entries to `None`; rows that share a name with the MultiMolecule vocab are copied directly.
    checkpoint_vocab_size = state_dict[embed_key].shape[0]
    new_vocab_set = set(new_vocab)
    aligned_original_vocab = [tok if tok in new_vocab_set else None for tok in original_vocab]
    while len(aligned_original_vocab) < checkpoint_vocab_size:
        aligned_original_vocab.append(None)

    word_embed_weight, decoder_weight = convert_word_embeddings(
        state_dict[embed_key],
        state_dict[decoder_key],
        old_vocab=aligned_original_vocab,
        new_vocab=new_vocab,
        std=config.initializer_range,
    )
    state_dict[embed_key] = word_embed_weight
    state_dict[decoder_key] = decoder_weight

    return state_dict


def convert_original_state_dict_key(key: str) -> str | None:
    """Map an upstream state-dict key to its MultiMolecule equivalent.

    Returns ``None`` for keys that should be dropped (e.g. non-persistent rotary buffers).
    """
    # Skip non-persistent buffers from the upstream rotary modules
    if key.endswith(".inv_freq") or key.endswith(".original_inv_freq"):
        return None

    if key.startswith("model.embed_tokens."):
        return "model.embeddings.word_embeddings." + key[len("model.embed_tokens.") :]
    if key.startswith("model.norm."):
        return "model.encoder.layer_norm." + key[len("model.norm.") :]
    if key == "lm_head.weight":
        return "lm_head.decoder.weight"

    if key.startswith("model.layers."):
        # model.layers.{i}.{rest} -> model.encoder.layer.{i}.{translated_rest}
        body = key[len("model.layers.") :]
        idx, _, rest = body.partition(".")
        rest = _translate_layer_subkey(rest)
        return f"model.encoder.layer.{idx}.{rest}"

    return key


def _translate_layer_subkey(rest: str) -> str:
    # input_layernorm -> input_layer_norm
    if rest.startswith("input_layernorm."):
        return "input_layer_norm." + rest[len("input_layernorm.") :]
    # post_attention_layernorm -> post_attention_layer_norm
    if rest.startswith("post_attention_layernorm."):
        return "post_attention_layer_norm." + rest[len("post_attention_layernorm.") :]
    # self_attn.q_proj/k_proj/v_proj/o_proj -> attention.{query,key,value,output}
    if rest.startswith("self_attn."):
        attn_rest = rest[len("self_attn.") :]
        return "attention." + _translate_attn_subkey(attn_rest)
    # mlp keeps its name (gate_proj, up_proj, down_proj)
    return rest


def _translate_attn_subkey(rest: str) -> str:
    if rest.startswith("q_proj."):
        return "query." + rest[len("q_proj.") :]
    if rest.startswith("k_proj."):
        return "key." + rest[len("k_proj.") :]
    if rest.startswith("v_proj."):
        return "value." + rest[len("v_proj.") :]
    if rest.startswith("o_proj."):
        return "output." + rest[len("o_proj.") :]
    return rest


_CHECKPOINT_NAME_MAP = {
    "GENERanno-eukaryote-0.5b-base": "generanno-eukaryote",
    "GENERanno-prokaryote-0.5b-base": "generanno-prokaryote",
}


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]
    default_variant: str | None = "generanno-eukaryote"

    def post(self):
        if self.output_path is None:
            basename = os.path.basename(self.checkpoint_path.rstrip("/"))
            self.output_path = _CHECKPOINT_NAME_MAP.get(basename, basename.lower())
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
