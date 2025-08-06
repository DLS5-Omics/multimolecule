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

import __main__
import chanfig
import torch

from multimolecule.models import RnaMsmConfig as Config
from multimolecule.models import RnaMsmForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet

torch.manual_seed(1016)

# evil hack
__main__.OptimizerConfig = chanfig.FlatDict()
__main__.MSATransformerModelConfig = chanfig.FlatDict()
__main__.DataConfig = chanfig.FlatDict()
__main__.TrainConfig = chanfig.FlatDict()
__main__.LoggingConfig = chanfig.FlatDict()


def convert_checkpoint(convert_config):
    vocab_list = get_alphabet().vocabulary
    config = Config(num_labels=1)
    config.architectures = ["RnaMsmModel"]
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    ckpt = ckpt.get("state_dict", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    load_checkpoint(model, state_dict)

    save_checkpoint(convert_config, model)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("layers", "rnamsm.encoder.layer")
        key = key.replace("msa_position_embedding", "rnamsm.embeddings.msa_embeddings")
        key = key.replace("embed_tokens", "rnamsm.embeddings.word_embeddings")
        key = key.replace("embed_positions", "rnamsm.embeddings.position_embeddings")
        key = key.replace("emb_layer_norm_before", "rnamsm.embeddings.layer_norm")
        key = key.replace("emb_layer_norm_after", "rnamsm.encoder.layer_norm")
        key = key.replace("regression", "decoder")
        key = key.replace("contact_head", "ss_head")
        key = key.replace("lm_head.weight", "lm_head.decoder.weight")
        key = key.replace("lm_head.dense", "lm_head.transform.dense")
        key = key.replace("lm_head.layer_norm", "lm_head.transform.layer_norm")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["rnamsm.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rnamsm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


original_vocab_list = ["<cls>", "<pad>", "<eos>", "<unk>", "A", "G", "C", "U", "X", "N", "-", "<mask>"]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
