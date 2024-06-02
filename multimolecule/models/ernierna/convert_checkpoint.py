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

import torch

from multimolecule.models import ErnieRnaConfig as Config
from multimolecule.models import ErnieRnaForContactClassification, ErnieRnaForPreTraining
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_vocab_list

torch.manual_seed(1013)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("module.sentence_encoder", "encoder")
        key = key.replace("encoder.sentence_encoder", "ernierna.encoder")
        key = key.replace("layers", "layer")
        key = key.replace("self_attn", "attention.self")
        key = key.replace("q_proj", "query")
        key = key.replace("k_proj", "key")
        key = key.replace("v_proj", "value")
        key = key.replace("self.out_proj", "output.dense")
        key = key.replace("self_layer_norm", "output.layer_norm")
        key = key.replace("final_layer_norm", "output.layer_norm")
        key = key.replace("fc1", "intermediate.dense")
        key = key.replace("fc2", "output.dense")
        key = key.replace("encoder.twod_proj.linear1", "ernierna.pairwise_bias_proj.0")
        key = key.replace("encoder.twod_proj.linear2", "ernierna.pairwise_bias_proj.2")
        key = key.replace("encoder.embed_tokens", "embeddings.word_embeddings")
        key = key.replace("encoder.embed_positions", "embeddings.position_embeddings")
        key = key.replace("encoder.segment_embeddings", "embeddings.token_type_embeddings")
        key = key.replace("encoder.emb_layer_norm", "embeddings.layer_norm")
        key = key.replace("encoder.lm_head_transform_weight", "lm_head.transform.dense")
        key = key.replace("encoder.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("encoder.masked_lm_pooler", "ernierna.pooler.dense")
        key = key.replace("encoder.lm_output_learned_bias", "lm_head.decoder.bias")
        key = key.replace("module", "ss_head")
        key = key.replace("proj.resnet", "resnet")
        key = key.replace("proj.final", "resnet.8")
        state_dict[key] = value

    state_dict.pop("ernierna.embeddings.position_embeddings._float_tensor", None)
    state_dict.pop("encoder.sentence_projection_layer.weight", None)

    word_embed_weight, decoder_bias = convert_word_embeddings(
        state_dict["ernierna.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["ernierna.embeddings.word_embeddings.weight"] = state_dict["lm_head.decoder.weight"] = word_embed_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    vocab_list = get_vocab_list()
    original_vocab_list = [
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "G",
        "A",
        "U",
        "C",
        "N",
        "Y",
        "R",
        "S",
        "K",
        "W",
        "M",
        "D",
        "H",
        "V",
        "B",
        "X",
        "I",
        "<null>",
        "<null>",
        "<null>",
        "<mask>",
    ]
    config = Config()
    config.architectures = ["ErnieRnaModel"]
    config.vocab_size = len(vocab_list)

    Model = ErnieRnaForPreTraining
    if "ss" in convert_config.checkpoint_path:
        Model = ErnieRnaForContactClassification

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)
    if config.position_embedding_type == "sinusoidal":
        state_dict["ernierna.embeddings.position_embeddings.weight"] = (
            model.ernierna.embeddings.position_embeddings.weight
        )

    model.load_state_dict(state_dict)

    save_checkpoint(convert_config, model)


@dataclass
class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
