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

from multimolecule.models import UtrBertConfig as Config
from multimolecule.models import UtrBertForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_tokenizer_config, get_vocab_list

torch.manual_seed(1013)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        if key.startswith("bert"):
            state_dict["utr" + key] = value
            continue
        if key.startswith("cls"):
            key = "lm_head" + key[15:]
            state_dict[key] = value
            continue
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["utrbert.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["utrbert.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config.hidden_dropout = config.pop("hidden_dropout_prob", 0.1)
    config.attention_dropout = config.pop("attention_probs_dropout_prob", 0.1)
    config.nmers = int(convert_config.checkpoint_path.split("/")[-1][0])
    vocab_list = get_vocab_list(nmers=config.nmers)
    config = Config.from_dict(config)
    del config._name_or_path
    config.architectures = ["UtrBertModel"]
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(
        os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
    )
    original_vocab_list = []
    for char in open(os.path.join(convert_config.checkpoint_path, "vocab.txt")).read().splitlines():  # noqa: SIM115
        if char.startswith("["):
            char = char.lower().replace("[", "<").replace("]", ">")
        if char == "T":
            char = "U"
        if char == "<sep>":
            char = "<eos>"
        original_vocab_list.append(char)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["nmers"] = config.nmers
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 2

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)


@dataclass
class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]

    def post(self):
        if self.output_path is None:
            self.output_path = self.checkpoint_path.split("/")[-1].lower()
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
