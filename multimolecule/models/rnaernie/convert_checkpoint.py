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

import torch

from multimolecule.models import RnaErnieConfig as Config
from multimolecule.models import RnaErnieForPreTraining
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet

torch.manual_seed(1016)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("ernie"):
            key = "rna" + key
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("cls", "bias")
        key = key.replace("layers", "layer")
        key = key.replace("self_attn", "attention.self")
        key = key.replace("self.out_proj", "output.dense")
        key = key.replace("q_proj", "query")
        key = key.replace("k_proj", "key")
        key = key.replace("v_proj", "value")
        key = key.replace("linear1", "intermediate.dense")
        key = key.replace("linear2", "output.dense")
        key = key.replace("norm1", "attention.output.layer_norm")
        key = key.replace("norm2", "output.layer_norm")
        key = key.replace("bias.predictions", "lm_head")
        key = key.replace("transform", "transform.dense")
        key = key.replace("lm_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("decoder_", "decoder.")
        state_dict[key] = value

    for key, value in state_dict.items():
        # Zhiyuan: Is this right? Why do we need to transpose these weights?
        # All other weights are not transposed
        if "output.dense.weight" in key or "intermediate.dense.weight" in key:
            state_dict[key] = value.t()

    word_embed_weight, decoder_bias = convert_word_embeddings(
        state_dict["rnaernie.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rnaernie.embeddings.word_embeddings.weight"] = state_dict["lm_head.decoder.weight"] = word_embed_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    vocab_list = get_alphabet().vocabulary
    original_vocab_list = [
        "<pad>",
        "<unk>",
        "<cls>",
        "<eos>",
        "<mask>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "A",
        "U",
        "C",
        "G",
    ]
    config = Config()
    config.architectures = ["RnaErnieModel"]
    config.vocab_size = len(vocab_list)

    model = RnaErnieForPreTraining(config)

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
