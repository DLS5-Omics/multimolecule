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

from multimolecule.models import UniRnaConfig as Config
from multimolecule.models import UniRnaForPreTraining, UniRnaForSecondaryStructurePrediction
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


def convert_checkpoint(convert_config):
    print(f"Converting UniRna checkpoint at {convert_config.checkpoint_path}")

    path = convert_config.checkpoint_path.lower()

    config = Config(num_labels=1)
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    vocab_list = get_alphabet().vocabulary
    original_vocab_list = original_vocab
    if convert_config.size == "l24":
        config = Config(num_hidden_layers=24, hidden_size=1280, num_attention_heads=20, intermediate_size=3840)
    elif convert_config.size == "l16":
        config = Config(num_hidden_layers=16, hidden_size=1024, num_attention_heads=16, intermediate_size=3072)
    elif convert_config.size == "l12":
        config = Config(num_hidden_layers=12, hidden_size=768, num_attention_heads=12, intermediate_size=2304)
    elif convert_config.size == "l8":
        config = Config(num_hidden_layers=8, hidden_size=512, num_attention_heads=8, intermediate_size=1536)
    if "ss" in path:
        Model = UniRnaForSecondaryStructurePrediction
        convert_config.output_path += "-ss"
        convert_config.repo_id += "-ss"
    else:
        Model = UniRnaForPreTraining
    config.vocab_size = len(vocab_list)
    config.architectures = ["UniRnaModel"]
    tokenizer_config["model_max_length"] = config.max_position_embeddings - 2

    model = Model(config)

    checkpoint_path = (
        os.path.join(convert_config.checkpoint_path, "pytorch_model.bin")
        if os.path.isdir(convert_config.checkpoint_path)
        else convert_config.checkpoint_path
    )

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        if "head" not in key:
            key = "unirna." + key
        if "inv_freq" in key:
            continue
        if "lm_head.dense" in key or "lm_head.layer_norm" in key:
            key = key.replace("lm_head.", "lm_head.transform.")
        key = key.replace("heads.label", "ss_head")
        key = key.replace("backbone.sequence.", "")
        key = key.replace("LayerNorm", "layer_norm")
        state_dict[key] = value

    if "lm_head.decoder.weight" in state_dict:
        word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
            state_dict["unirna.embeddings.word_embeddings.weight"],
            state_dict["lm_head.decoder.weight"],
            state_dict["lm_head.decoder.bias"],
            old_vocab=original_vocab_list,
            new_vocab=vocab_list,
            std=config.initializer_range,
        )
        state_dict["unirna.embeddings.word_embeddings.weight"] = word_embed_weight
        state_dict["lm_head.decoder.weight"] = decoder_weight
        state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    else:
        state_dict["unirna.embeddings.word_embeddings.weight"] = convert_word_embeddings(
            state_dict["unirna.embeddings.word_embeddings.weight"],
            old_vocab=original_vocab_list,
            new_vocab=vocab_list,
            std=config.initializer_range,
        )[0]
    return state_dict


original_vocab = [
    "<pad>",
    "<eos>",
    "N",
    "<cls>",
    "<mask>",
    "A",
    "U",
    "C",
    "G",
    "<null>",
]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    size: str | None = None

    def post(self):
        if self.size is None:
            if "l24" in self.checkpoint_path.lower():
                self.size = "l24"
            elif "l16" in self.checkpoint_path.lower():
                self.size = "l16"
            elif "l12" in self.checkpoint_path.lower():
                self.size = "l12"
            elif "l8" in self.checkpoint_path.lower():
                self.size = "l8"
            else:
                self.size = "l16"
        self.output_path += f"-{self.size}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
