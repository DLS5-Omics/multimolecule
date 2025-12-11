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

from multimolecule.models import RibonanzaNetConfig as Config
from multimolecule.models import (
    RibonanzaNetForDegradationPrediction,
    RibonanzaNetForPreTraining,
    RibonanzaNetForSecondaryStructurePrediction,
    RibonanzaNetForSequenceDropoutPrediction,
)
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet

torch.manual_seed(1016)


def convert_checkpoint(convert_config):
    print(f"Converting RibonanzaNet checkpoint at {convert_config.checkpoint_path}")
    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)

    config = Config()
    config.use_triangular_attention = any("triangle_attention" in key for key in ckpt.keys())
    vocab_list = get_alphabet().vocabulary
    config.vocab_size = len(vocab_list)
    config.architectures = ["RibonanzaNetModel"]

    if "ct_predictor.weight" in ckpt:
        Model = RibonanzaNetForSecondaryStructurePrediction
        convert_config.output_path += "-ss"
        convert_config.repo_id += "-ss"
    elif "Deg" in convert_config.checkpoint_path:
        Model = RibonanzaNetForDegradationPrediction
        convert_config.output_path += "-deg"
        convert_config.repo_id += "-deg"
    elif "Drop" in convert_config.checkpoint_path:
        Model = RibonanzaNetForSequenceDropoutPrediction
        convert_config.output_path += "-drop"
        convert_config.repo_id += "-drop"
    else:
        Model = RibonanzaNetForPreTraining

    model = Model(config)

    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("model.", "")
        if key.startswith("encoder"):
            key = key.replace("encoder", "model.embeddings.word_embeddings")
        key = key.replace("pos_encoder.linear", "model.encoder.pairwise_embeddings.position_embeddings")
        key = key.replace("transformer_encoder", "model.encoder.layers")
        key = key.replace("triangle", "pairwise.triangle")
        if key.startswith("outer_product_mean"):
            key = key.replace("outer_product_mean", "model.encoder.pairwise_embeddings.triangle_proj")
        else:
            key = key.replace("outer_product_mean", "pairwise.triangle_proj")
        key = key.replace("pairwise_norm", "pairwise_to_bias.0")
        key = key.replace("pairwise2heads", "pairwise_to_bias.1")
        key = key.replace("norm3", "conv_norm")
        key = key.replace("norm2", "output.layer_norm")
        key = key.replace("norm1", "attention.output.layer_norm2")
        key = key.replace("self_attn.w_qs", "attention.self.query")
        key = key.replace("self_attn.w_ks", "attention.self.key")
        key = key.replace("self_attn.w_vs", "attention.self.value")
        key = key.replace("self_attn.fc", "attention.output.dense")
        key = key.replace("self_attn.layer_norm", "attention.output.layer_norm")
        key = key.replace("linear1", "intermediate.dense")
        key = key.replace("linear2", "output.dense")
        key = key.replace(".norm", ".layer_norm")
        key = key.replace("to_gate", "gate")
        key = key.replace("to_out_norm", "out_norm")
        key = key.replace("to_out", "out_proj")
        key = key.replace("proj_down1", "in_proj")
        key = key.replace("proj_down2", "out_proj")
        key = key.replace("linear_for_pair", "pairwise_bias")
        key = key.replace("pairwise_to_bias", "pairwise_bias")
        if "pair_transition" in key:
            key = key.replace("pair_transition", "pairwise")
            key = key.replace("pairwise.0", "pairwise.intermediate.layer_norm")
            key = key.replace("pairwise.1", "pairwise.intermediate.dense")
            key = key.replace("pairwise.3", "pairwise.output.dense")
        if "triangle_update_" in key:
            key = key.replace("triangle_update_", "triangle_mixer_")
            key = key.replace(".layer_norm", ".in_norm")
        if "triangle_attention_" in key:
            key = key.replace("triangle_attention_in", "triangle_attention_in.triangle")
            key = key.replace("triangle_attention_out", "triangle_attention_out.triangle")
            key = key.replace("triangle.out_proj", "output.dense")
        key = key.replace("ct_predictor", "ss_head.decoder")
        key = key.replace("head.layers.0", "decoder")
        if "to_qkv" in key:
            q, k, v = (
                key.replace("to_qkv", "query"),
                key.replace("to_qkv", "key"),
                key.replace("to_qkv", "value"),
            )
            state_dict[q], state_dict[k], state_dict[v] = value.chunk(3, dim=0)
        if "decoder" in key:
            if value.shape[0] == 2:
                a3c, dms = key.replace("decoder", "a3c_head.decoder"), key.replace("decoder", "dms_head.decoder")
                state_dict[a3c], state_dict[dms] = value[:, None]
            elif value.shape[0] == 5:
                reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C = (
                    key.replace("decoder", "reactivity_head.decoder"),
                    key.replace("decoder", "deg_Mg_pH10_head.decoder"),
                    key.replace("decoder", "deg_pH10_head.decoder"),
                    key.replace("decoder", "deg_Mg_50C_head.decoder"),
                    key.replace("decoder", "deg_50C_head.decoder"),
                )
                (
                    state_dict[reactivity],
                    state_dict[deg_Mg_pH10],
                    state_dict[deg_pH10],
                    state_dict[deg_Mg_50C],
                    state_dict[deg_50C],
                ) = value[:, None]
            else:
                state_dict[key] = value
        else:
            state_dict[key] = value

    word_embed_weight = convert_word_embeddings(
        state_dict["model.embeddings.word_embeddings.weight"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )[0]
    word_embed_weight[0] = 0
    state_dict["model.embeddings.word_embeddings.weight"] = word_embed_weight
    return state_dict


original_vocab_list = ["A", "C", "G", "U", "N"]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
