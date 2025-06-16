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

from multimolecule.models import NcRnaBertConfig as Config
from multimolecule.models import NcRnaBertForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("embed_tokens", "ncrnabert.embeddings.word_embeddings")
        key = key.replace("layers.", "ncrnabert.encoder.layer.")
        key = key.replace("mha_layer", "attention")
        key = key.replace("rotary_emb", "self.rotary_embeddings")
        key = key.replace("q_proj", "self.query")
        key = key.replace("k_proj", "self.key")
        key = key.replace("v_proj", "self.value")
        key = key.replace("out_proj", "output.dense")
        key = key.replace("ffn.fc1", "intermediate.dense")
        key = key.replace("ffn.fc2", "output.dense")
        key = key.replace("self_attn_layer_norm", "attention.layer_norm")
        key = key.replace("final_layer_norm", "layer_norm")
        key = key.replace("emb_layer_norm_after", "ncrnabert.encoder.layer_norm")
        key = key.replace("lm_head.dense", "lm_head.transform.dense")
        key = key.replace("lm_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("lm_head.weight", "lm_head.decoder.weight")
        key = key.replace("lm_head.bias", "lm_head.decoder.bias")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["ncrnabert.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["ncrnabert.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias

    return state_dict


def convert_checkpoint(convert_config):
    nucleotides = "GAUC"

    config = Config()
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())

    if "3kmer" in convert_config.checkpoint_path:
        config = Config(nmers=3)
        tokenizer_config["nmers"] = config.nmers
        vocab_list = get_alphabet(nmers=config.nmers).vocabulary
        rna_3 = [i + j + k for i in nucleotides for j in nucleotides for k in nucleotides]
        rna_2 = ["<null>" for _ in nucleotides for _ in nucleotides]
        rna_1 = ["<null>" for _ in nucleotides]
        special_rna_tokens = ["<unk>", "<mask>", "<pad>"]
        original_vocab_list = rna_3 + rna_2 + rna_1 + special_rna_tokens
        convert_config.output_path += "-3mer"
        convert_config.repo_id += "-3mer"
    else:
        vocab_list = get_alphabet().vocabulary
        rna_3 = ["<null>" for _ in nucleotides for _ in nucleotides for _ in nucleotides]
        rna_2 = ["<null>" for _ in nucleotides for _ in nucleotides]
        rna_1 = list(nucleotides)
        special_rna_tokens = ["N", "<mask>", "<pad>"]
        original_vocab_list = rna_3 + rna_2 + rna_1 + special_rna_tokens
    tokenizer_config["model_max_length"] = config.max_position_embeddings

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))

    model = Model(config)
    state_dict = _convert_checkpoint(config, ckpt["state_dict"], vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
