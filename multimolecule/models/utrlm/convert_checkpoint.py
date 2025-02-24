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

from multimolecule.models import UtrLmConfig as Config
from multimolecule.models import UtrLmForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet

torch.manual_seed(1016)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        key = "utrlm." + key
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("utrlm.encoder.emb_layer_norm_before", "utrlm.embeddings.layer_norm")
        key = key.replace("utrlm.emb_layer_norm_after", "utrlm.encoder.emb_layer_norm_after")
        key = key.replace("utrlm.embed_tokens", "utrlm.embeddings.word_embeddings")
        key = key.replace("rot_emb", "rotary_embeddings")
        key = key.replace("layers", "encoder.layer")
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
        key = key.replace("utrlm.lm_head", "lm_head")
        key = key.replace("lm_head.dense", "lm_head.transform.dense")
        key = key.replace("lm_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("lm_head.weight", "lm_head.decoder.weight")
        key = key.replace("utrlm.contact_head", "ss_head")
        key = key.replace("utrlm.structure_linear", "structure_head.decoder")
        key = key.replace("utrlm.supervised_linear", "mfe_head.decoder")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["utrlm.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["utrlm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    config = chanfig.FlatDict(num_labels=1)
    config.mfe_head = {"num_labels": 1}
    if "4.1" in convert_config.checkpoint_path:
        config.structure_head = {"num_labels": 3}
    vocab_list = get_alphabet().vocabulary
    original_vocab_list = ["<pad>", "<eos>", "<unk>", "A", "G", "C", "U", "<cls>", "<mask>", "<eos>"]
    config = Config.from_dict(config)
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    save_checkpoint(convert_config, model)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
