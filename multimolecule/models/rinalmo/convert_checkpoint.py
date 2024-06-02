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
import torch

from multimolecule.models import RiNALMoConfig as Config
from multimolecule.models import RiNALMoForPreTraining as Model
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
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("embedding", "rinalmo.embeddings.word_embeddings")
        key = key.replace("transformer", "rinalmo")
        key = key.replace("blocks", "encoder.layer")
        key = key.replace("mh_attn", "attention")
        key = key.replace("attn_layer_norm", "attention.layer_norm")
        key = key.replace("out_proj", "output.dense")
        key = key.replace("out_layer_norm", "layer_norm")
        key = key.replace("transition.0", "intermediate")
        key = key.replace("transition.2", "output.dense")
        key = key.replace("final_layer_norm", "encoder.layer_norm")
        key = key.replace("lm_mask_head.linear1", "lm_head.transform.dense")
        key = key.replace("lm_mask_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("lm_mask_head.linear2", "lm_head.decoder")
        if "Wqkv" in key:
            q, k, v = (
                key.replace("Wqkv", "self.query"),
                key.replace("Wqkv", "self.key"),
                key.replace("Wqkv", "self.value"),
            )
            state_dict[q], state_dict[k], state_dict[v] = value.chunk(3, dim=0)
        else:
            state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["rinalmo.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rinalmo.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


original_vocab_list = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "<mask>",
    "A",
    "C",
    "G",
    "U",
    "I",
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
]


def convert_checkpoint(convert_config):
    config = Config()
    vocab_list = get_vocab_list()
    config.vocab_size = len(vocab_list)
    config.architectures = ["RnaFmModel"]

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)
    for key, value in model.state_dict().items():
        if "inv_freq" in key:
            state_dict[key] = value

    model.load_state_dict(state_dict)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 2

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
