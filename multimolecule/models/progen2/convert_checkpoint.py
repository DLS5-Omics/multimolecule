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

from multimolecule.models import ProGen2Config as Config
from multimolecule.models import ProGen2ForPreTraining as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.protein.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

QKV_MP_NUM = 8


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting ProGen2 checkpoint at {convert_config.checkpoint_path}")

    vocab_list = get_alphabet().vocabulary

    config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    orig_vocab_size = config.get("vocab_size", 32)
    config.vocab_size = len(vocab_list)
    config.embedding_dropout = config.pop("embd_pdrop", 0.0)
    config.hidden_dropout = config.pop("resid_pdrop", 0.0)
    config.attention_dropout = config.pop("attn_pdrop", 0.0)
    config.hidden_size = config.pop("n_embd", 4096)
    config.num_hidden_layers = config.pop("n_layer", 28)
    config.num_attention_heads = config.pop("n_head", 16)
    config.intermediate_size = config.pop("n_inner", 4 * config.hidden_size)
    config.hidden_act = config.pop("activation_function", "gelu_new")
    config.max_position_embeddings = config.pop("n_positions", 2048)
    config.rotary_dim = config.pop("rotary_dim", 64)
    config.is_decoder = config.pop("is_decoder", True)
    config.layer_norm_eps = config.pop("layer_norm_epsilon", 1e-5)
    config.gradient_checkpointing = config.pop("gradient_checkpointing", False)
    config = Config.from_dict(config)
    del config._name_or_path
    config.architectures = ["ProGen2Model"]
    config.tie_word_embeddings = False

    ckpt = torch.load(
        os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
    )

    orig_vocab = list(original_vocab_list)
    if orig_vocab_size > len(orig_vocab):
        orig_vocab.extend([None] * (orig_vocab_size - len(orig_vocab)))  # type: ignore[list-item]
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, orig_vocab)

    model = Model(config)

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=get_tokenizer_config())
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("transformer.wte", "model.embeddings.word_embeddings")
        key = key.replace("transformer.h", "model.decoder.layers")
        key = key.replace("transformer.ln_f", "model.layer_norm")
        key = key.replace(".attn", ".attention")
        key = key.replace(".ln_1", ".layer_norm")
        # Skip causal mask buffers (no longer stored as model state)
        if key.endswith("attention.bias") or key.endswith("attention.masked_bias"):
            continue
        if "qkv_proj" in key:
            q, v, k = (
                key.replace("qkv_proj", "q_proj"),
                key.replace("qkv_proj", "v_proj"),
                key.replace("qkv_proj", "k_proj"),
            )
            if key.endswith(".bias"):
                q_bias, v_bias, k_bias = _split_qkv_bias(value, config.hidden_size)
                state_dict[q] = q_bias
                state_dict[v] = v_bias
                state_dict[k] = k_bias
            else:
                q_weight, v_weight, k_weight = _split_qkv_weight(value, config.hidden_size)
                state_dict[q] = q_weight
                state_dict[v] = v_weight
                state_dict[k] = k_weight
        else:
            state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["model.embeddings.word_embeddings.weight"],
        state_dict["lm_head.weight"],
        state_dict["lm_head.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.weight"] = decoder_weight
    state_dict["lm_head.bias"] = decoder_bias

    return state_dict


def _split_qkv_weight(weight: torch.Tensor, hidden_size: int, mp_num: int = QKV_MP_NUM):
    if weight.shape[0] != hidden_size * 3:
        raise ValueError(f"Unexpected qkv weight shape {weight.shape} for hidden size {hidden_size}.")
    if hidden_size % mp_num != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by mp_num={mp_num}.")
    local_dim = hidden_size // mp_num
    reshaped = weight.view(mp_num, local_dim * 3, hidden_size)
    q = reshaped[:, :local_dim, :].reshape(hidden_size, hidden_size)
    v = reshaped[:, local_dim : local_dim * 2, :].reshape(hidden_size, hidden_size)
    k = reshaped[:, local_dim * 2 :, :].reshape(hidden_size, hidden_size)
    return q.contiguous(), v.contiguous(), k.contiguous()


def _split_qkv_bias(bias: torch.Tensor, hidden_size: int, mp_num: int = QKV_MP_NUM):
    if bias.shape[0] != hidden_size * 3:
        raise ValueError(f"Unexpected qkv bias shape {bias.shape} for hidden size {hidden_size}.")
    if hidden_size % mp_num != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by mp_num={mp_num}.")
    local_dim = hidden_size // mp_num
    reshaped = bias.view(mp_num, local_dim * 3)
    q = reshaped[:, :local_dim].reshape(hidden_size)
    v = reshaped[:, local_dim : local_dim * 2].reshape(hidden_size)
    k = reshaped[:, local_dim * 2 :].reshape(hidden_size)
    return q.contiguous(), v.contiguous(), k.contiguous()


original_vocab_list = [
    "<pad>",
    "<cls>",
    "<eos>",
    "<null>",
    "<null>",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<null>",
    "<null>",
]


class ConvertConfig(ConvertConfig_):
    fp16: bool = True
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "progen2-base"

    def post(self):
        # Determine model variant from checkpoint path
        checkpoint_path = self.checkpoint_path.lower()
        if "xlarge" in checkpoint_path:
            self.output_path += "-xlarge"
        elif "large" in checkpoint_path:
            self.output_path += "-large"
        elif "medium" in checkpoint_path:
            self.output_path += "-medium"
        elif "small" in checkpoint_path:
            self.output_path += "-small"
        elif "bfd90" in checkpoint_path:
            self.output_path += "-bfd90"
        elif "oas" in checkpoint_path:
            self.output_path += "-oas"
        elif "base" in checkpoint_path:
            self.output_path += "-base"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
