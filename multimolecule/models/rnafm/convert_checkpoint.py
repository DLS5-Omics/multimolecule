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
from copy import deepcopy
from dataclasses import dataclass

import chanfig
import torch

from multimolecule.models import RnaFmConfig as Config
from multimolecule.models import RnaFmForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import (
    convert_word_embeddings,
    get_special_tokens_map,
    get_tokenizer_config,
    get_vocab_list,
)

torch.manual_seed(1013)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = "rnafm" + key[7:]
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("rnafm.encoder.emb_layer_norm_before", "rnafm.embeddings.layer_norm")
        key = key.replace("rnafm.encoder.embed_tokens", "rnafm.embeddings.word_embeddings")
        key = key.replace("rnafm.encoder.embed_positions", "rnafm.embeddings.position_embeddings")
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
        key = key.replace("regression", "decoder")
        key = key.replace("rnafm.encoder.lm_head", "pretrain.predictions")
        key = key.replace("predictions.dense", "predictions.transform.dense")
        key = key.replace("predictions.layer_norm", "predictions.transform.layer_norm")
        key = key.replace("predictions.weight", "predictions.decoder.weight")
        key = key.replace("rnafm.encoder.contact_head", "pretrain.contact_head")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["rnafm.embeddings.word_embeddings.weight"],
        state_dict["pretrain.predictions.decoder.weight"],
        state_dict["pretrain.predictions.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rnafm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["pretrain.predictions.decoder.weight"] = decoder_weight
    state_dict["pretrain.predictions.decoder.bias"] = state_dict["pretrain.predictions.bias"] = decoder_bias
    return state_dict


original_vocabs = {
    "single": [
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "A",
        "C",
        "G",
        "U",
        "R",
        "Y",
        "K",
        "M",
        "S",
        "W",
        "B",
        "D",
        "H",
        "V",
        "N",
        "-",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<mask>",
    ],
    "3mer": [
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "GAG",
        "AAG",
        "GAA",
        "CUG",
        "CAG",
        "GAU",
        "AAA",
        "GUG",
        "GAC",
        "AUG",
        "GCC",
        "AAC",
        "GCU",
        "AAU",
        "AUC",
        "UUC",
        "GGA",
        "AUU",
        "GGC",
        "UUU",
        "CCA",
        "AGC",
        "GCA",
        "UCU",
        "CUC",
        "ACC",
        "CAA",
        "CCU",
        "UCC",
        "ACA",
        "UUG",
        "GUU",
        "CUU",
        "UAC",
        "ACU",
        "CCC",
        "UCA",
        "GUC",
        "GGU",
        "CAC",
        "AGU",
        "UAU",
        "AGA",
        "CAU",
        "GGG",
        "UGG",
        "UGC",
        "AGG",
        "UGU",
        "AUA",
        "CGC",
        "UUA",
        "GCG",
        "CGG",
        "CCG",
        "GUA",
        "CUA",
        "ACG",
        "UCG",
        "CGA",
        "CGU",
        "UGA",
        "UAA",
        "UAG",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<mask>",
    ],
}


def convert_checkpoint(convert_config):
    mranfm = not convert_config.output_path.lower().startswith("rnafm")
    if mranfm:
        config = Config(num_labels=1, hidden_size=1280, emb_layer_norm_before=False)
        vocab_list = get_vocab_list(nmers=3)
        original_vocab_list = original_vocabs["3mer"]
    else:
        config = Config(num_labels=1)
        config.codon = True
        vocab_list = get_vocab_list()
        original_vocab_list = original_vocabs["single"]
    config.vocab_size = len(vocab_list)
    config.architectures = ["RnaFmModel"]

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    model.lm_head = deepcopy(model.pretrain.predictions)
    model.contact_head = deepcopy(model.pretrain.contact_head)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 2
    if mranfm:
        tokenizer_config["codon"] = True

    save_checkpoint(
        convert_config, model, tokenizer_config=tokenizer_config, special_tokens_map=get_special_tokens_map()
    )


@dataclass
class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
