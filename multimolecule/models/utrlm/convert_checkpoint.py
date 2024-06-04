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

from multimolecule.models import UtrLmConfig as Config
from multimolecule.models import UtrLmForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_tokenizer_config, get_vocab_list

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

torch.manual_seed(1013)


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
        key = key.replace("utrlm.lm_head", "pretrain.predictions")
        key = key.replace("predictions.dense", "predictions.transform.dense")
        key = key.replace("predictions.layer_norm", "predictions.transform.layer_norm")
        key = key.replace("predictions.weight", "predictions.decoder.weight")
        key = key.replace("utrlm.contact_head", "pretrain.contact_head")
        key = key.replace("utrlm.structure_linear", "pretrain.ss_head.decoder")
        key = key.replace("utrlm.supervised_linear", "pretrain.mfe_head.decoder")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["utrlm.embeddings.word_embeddings.weight"],
        state_dict["pretrain.predictions.decoder.weight"],
        state_dict["pretrain.predictions.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["utrlm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["pretrain.predictions.decoder.weight"] = decoder_weight
    state_dict["pretrain.predictions.decoder.bias"] = state_dict["pretrain.predictions.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    config = chanfig.FlatDict(num_labels=1)
    config.mfe_head = {"num_labels": 1}
    if "4.1" in convert_config.checkpoint_path:
        config.ss_head = {"num_labels": 3}
    vocab_list = get_vocab_list()
    original_vocab_list = ["<pad>", "<eos>", "<unk>", "A", "G", "C", "U", "<cls>", "<mask>", "<eos>"]
    config = Config.from_dict(config)
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    model.lm_head = deepcopy(model.pretrain.predictions)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["model_max_length"] = 1022

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)


@dataclass
class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
