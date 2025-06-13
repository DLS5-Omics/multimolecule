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

import torch

from multimolecule.models import AidoRnaConfig as Config
from multimolecule.models import AidoRnaForPreTraining, AidoRnaForSecondaryStructurePrediction
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet

torch.manual_seed(1016)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        if key == "threshold":
            continue
        key = key.replace("bert", "aido_rna")
        key = key.replace("backbone.encoder", "aido_rna")
        key = key.replace("ln", "layer_norm")
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("mlp.gate_proj", "intermediate.dense")
        key = key.replace("mlp.up_proj", "intermediate.gate")
        key = key.replace("mlp.down_proj", "output.dense")
        key = key.replace("adapter", "ss_head")
        key = key.replace("in_to_conv", "projection")
        key = key.replace("resnet", "convnet")
        key = key.replace("cls.predictions", "lm_head")
        key = key.replace("conv_to_output", "prediction")
        key = key.replace("res2D_blocks.", "")
        key = key.replace("residual_block.0", "conv1")
        key = key.replace("residual_block.3", "conv2")
        key = key.replace("residual_block.6", "conv3")
        state_dict[key] = value

    if "threshold" in original_state_dict:
        state_dict["aido_rna.embeddings.word_embeddings.weight"] = convert_word_embeddings(
            state_dict["aido_rna.embeddings.word_embeddings.weight"],
            old_vocab=original_vocab_list,
            new_vocab=vocab_list,
            std=config.initializer_range,
        )[0]
    else:
        word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
            state_dict["aido_rna.embeddings.word_embeddings.weight"],
            state_dict["lm_head.decoder.weight"],
            state_dict["lm_head.decoder.bias"],
            old_vocab=original_vocab_list,
            new_vocab=vocab_list,
            std=config.initializer_range,
        )
        state_dict["aido_rna.embeddings.word_embeddings.weight"] = word_embed_weight
        state_dict["lm_head.decoder.weight"] = decoder_weight
        state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias

    return state_dict


def convert_checkpoint(convert_config):
    vocab_list = get_alphabet().vocabulary
    original_vocab_list = [
        "<pad>",
        "<mask>",
        "<cls>",
        "<eos>",
        "<unk>",
        "A",
        "G",
        "C",
        "<null>",
        "U",
        "N",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
    ]

    if "AIDO.RNA-1.6B" in convert_config.checkpoint_path:
        config = Config()
        convert_config.output_path += "-1.6b"
        convert_config.repo_id += "-1.6b"
    elif "AIDO.RNA-650M" in convert_config.checkpoint_path:
        config = Config(num_hidden_layers=33, hidden_size=1280, num_attention_heads=20, intermediate_size=3392)
        convert_config.output_path += "-650m"
        convert_config.repo_id += "-650m"
    else:
        raise ValueError("Unable to determine model configuration.")

    if os.path.isfile(convert_config.checkpoint_path):
        if "secondary_structure" in convert_config.checkpoint_path:
            Model = AidoRnaForSecondaryStructurePrediction
            ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
            state_dict = ckpt["state_dict"]
            convert_config.output_path += "-ss"
            convert_config.repo_id += "-ss"
        else:
            raise ValueError("Unable to determine model type from checkpoint path.")
    else:
        if convert_config.checkpoint_path.endswith("-CDS"):
            convert_config.output_path += "-cds"
            convert_config.repo_id += "-cds"
        Model = AidoRnaForPreTraining
        files = [i for i in os.listdir(convert_config.checkpoint_path) if i.endswith(".bin")]
        state_dict = {}
        for file in files:
            ckpt = torch.load(os.path.join(convert_config.checkpoint_path, file), map_location=torch.device("cpu"))
            if "state_dict" in ckpt:
                state_dict.update(ckpt["state_dict"])
            else:
                state_dict.update(ckpt)

    model = Model(config)
    state_dict = _convert_checkpoint(config, state_dict, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)

    save_checkpoint(convert_config, model)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
