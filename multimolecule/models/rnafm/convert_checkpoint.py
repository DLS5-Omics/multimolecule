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

import chanfig
import torch

from multimolecule.models import RnaFmConfig as Config
from multimolecule.models import RnaFmForPreTraining, RnaFmForSecondaryStructurePrediction
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


def convert_checkpoint(convert_config):
    print(f"Converting RnaFm checkpoint at {convert_config.checkpoint_path}")

    path = convert_config.checkpoint_path.lower()

    config = Config(num_labels=1)
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    vocab_list = get_alphabet().vocabulary
    original_vocab_list = original_vocabs["single"]
    if "mrna" in path or "cds" in path:
        Model = RnaFmForPreTraining
        config = Config(num_labels=1, hidden_size=1280, embed_norm=False)
        vocab_list = get_alphabet(nmers=3).vocabulary
        original_vocab_list = original_vocabs["3mer"]
        convert_config.output_path = "mrnafm"
        convert_config.repo_id = "multimolecule/mrnafm"
        config.codon = True
        tokenizer_config["codon"] = True
    elif "resnet" in path:
        Model = RnaFmForSecondaryStructurePrediction
        convert_config.output_path += "-ss"
        convert_config.repo_id += "-ss"
    else:
        Model = RnaFmForPreTraining
    config.vocab_size = len(vocab_list)
    config.architectures = ["RnaFmModel"]
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 2

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("encoder.encoder", "rnafm.encoder")
        key = key.replace("backbone", "rnafm.encoder")
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("emb_layer_norm_after", "layer_norm")
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
        key = key.replace("rnafm.encoder.lm_head", "lm_head")
        key = key.replace("lm_head.dense", "lm_head.transform.dense")
        key = key.replace("lm_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("lm_head.weight", "lm_head.decoder.weight")
        key = key.replace("rnafm.encoder.contact_head", "ss_head")
        key = key.replace("downstream_modules.pc-resnet_1_sym_first:r-ss", "ss_head")
        key = key.replace("pre_reduction", "reduction")
        key = key.replace("proj.first.0", "projection")
        key = key.replace("proj.resnet", "convnet")
        key = key.replace("proj.final", "prediction")
        state_dict[key] = value

    if "ss_head.prediction.weight" in state_dict:
        state_dict.pop("ss_head.decoder.weight", None)
        state_dict.pop("ss_head.decoder.bias", None)

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["rnafm.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rnafm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
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


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
