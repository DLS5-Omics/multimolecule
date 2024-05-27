# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import os
from dataclasses import dataclass

import chanfig
import danling as dl
import torch

from multimolecule.models import CaLmConfig as Config
from multimolecule.models import CaLmForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_tokenizer_config, get_vocab_list

torch.manual_seed(1013)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        if "lm_head" not in key and "embed" not in key:
            key = "calm.encoder." + key
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("layers", "layer")
        key = key.replace("self_attn", "attention.self")
        key = key.replace("q_proj", "query")
        key = key.replace("k_proj", "key")
        key = key.replace("v_proj", "value")
        key = key.replace("self.out_proj", "output.dense")
        key = key.replace("self_layer_norm", "layer_norm")
        key = key.replace("final_layer_norm", "layer_norm")
        key = key.replace("fc1", "intermediate.dense")
        key = key.replace("fc2", "output.dense")
        key = key.replace("rope.freqs", "rotary_embeddings.inv_freq")
        key = key.replace("embed_tokens", "calm.embeddings.word_embeddings")
        key = key.replace("lm_head.dense", "lm_head.transform.dense")
        key = key.replace("lm_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("lm_head.weight", "lm_head.decoder.weight")
        key = key.replace("lm_head.bias", "lm_head.decoder.bias")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["calm.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["calm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


original_vocab = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "AAA",
    "AAU",
    "AAC",
    "AAG",
    "AUA",
    "AUU",
    "AUC",
    "AUG",
    "ACA",
    "ACU",
    "ACC",
    "ACG",
    "AGA",
    "AGU",
    "AGC",
    "AGG",
    "UAA",
    "UAU",
    "UAC",
    "UAG",
    "UUA",
    "UUU",
    "UUC",
    "UUG",
    "UCA",
    "UCU",
    "UCC",
    "UCG",
    "UGA",
    "UGU",
    "UGC",
    "UGG",
    "CAA",
    "CAU",
    "CAC",
    "CAG",
    "CUA",
    "CUU",
    "CUC",
    "CUG",
    "CCA",
    "CCU",
    "CCC",
    "CCG",
    "CGA",
    "CGU",
    "CGC",
    "CGG",
    "GAA",
    "GAU",
    "GAC",
    "GAG",
    "GUA",
    "GUU",
    "GUC",
    "GUG",
    "GCA",
    "GCU",
    "GCC",
    "GCG",
    "GGA",
    "GGU",
    "GGC",
    "GGG",
    "<mask>",
]


def convert_checkpoint(convert_config):
    original_vocab_list = original_vocab
    vocab_list = get_vocab_list(nmers=3)
    config = Config()
    del config._name_or_path
    config.architectures = ["CaLmModel"]
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = dl.load(convert_config.checkpoint_path)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 2
    tokenizer_config["codon"] = True

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)


@dataclass
class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
