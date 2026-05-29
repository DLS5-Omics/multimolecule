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
import pickle

import torch

from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.models.proteinbert.configuration_proteinbert import ProteinBertConfig as Config
from multimolecule.models.proteinbert.modeling_proteinbert import ProteinBertForPreTraining as Model
from multimolecule.tokenisers.protein.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

ORIGINAL_VOCAB_LIST: list[str] = [
    "A",
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
    "<unk>",
    "<cls>",
    "<eos>",
    "<pad>",
]


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting ProteinBERT checkpoint at {convert_config.checkpoint_path}")
    vocab_list = list(get_alphabet().vocabulary)

    annotation_size, weights = _load_dump(convert_config.checkpoint_path)
    config = Config(vocab_size=len(vocab_list), annotation_size=annotation_size)
    config.architectures = ["ProteinBertForPreTraining"]

    state_dict = _convert_checkpoint(config, weights, vocab_list, ORIGINAL_VOCAB_LIST)

    model = Model(config)
    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=get_tokenizer_config())
    print(f"Checkpoint saved to {convert_config.output_path}")


def _load_dump(checkpoint_path: str) -> tuple[int, list[torch.Tensor]]:
    with open(checkpoint_path, "rb") as f:
        annotation_size, weights, _ = pickle.load(f)
    return int(annotation_size), [torch.as_tensor(weight) for weight in weights]


def _convert_checkpoint(
    config: Config,
    weights: list[torch.Tensor],
    vocab_list: list[str],
    original_vocab_list: list[str],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    state_dict["model.embeddings.annotation_embeddings.weight"] = weights[0].T.contiguous()
    state_dict["model.embeddings.annotation_embeddings.bias"] = weights[1].contiguous()

    word_embeddings = weights[2]
    decoder_weight = weights[141].T.contiguous()
    decoder_bias = weights[142].contiguous()
    word_embeddings, decoder_weight, decoder_bias = convert_word_embeddings(
        word_embeddings,
        decoder_weight,
        decoder_bias,
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embeddings
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias

    state_dict["annotation_head.decoder.weight"] = weights[143].T.contiguous()
    state_dict["annotation_head.decoder.bias"] = weights[144].contiguous()

    for layer_idx in range(config.num_hidden_layers):
        offset = 3 + layer_idx * 23
        prefix = f"model.encoder.layer.{layer_idx}"
        state_dict[f"{prefix}.global_to_sequence.weight"] = weights[offset].T.contiguous()
        state_dict[f"{prefix}.global_to_sequence.bias"] = weights[offset + 1].contiguous()
        _convert_conv_branch(state_dict, f"{prefix}.narrow_conv.conv", weights[offset + 2], weights[offset + 3])
        _convert_conv_branch(state_dict, f"{prefix}.wide_conv.conv", weights[offset + 4], weights[offset + 5])
        _copy_layer_norm(state_dict, f"{prefix}.sequence_layer_norm1", weights[offset + 6], weights[offset + 7])
        state_dict[f"{prefix}.sequence_dense.weight"] = weights[offset + 8].T.contiguous()
        state_dict[f"{prefix}.sequence_dense.bias"] = weights[offset + 9].contiguous()
        _copy_layer_norm(state_dict, f"{prefix}.sequence_layer_norm2", weights[offset + 10], weights[offset + 11])
        state_dict[f"{prefix}.global_dense1.weight"] = weights[offset + 12].T.contiguous()
        state_dict[f"{prefix}.global_dense1.bias"] = weights[offset + 13].contiguous()
        state_dict[f"{prefix}.global_attention.query"] = weights[offset + 14].contiguous()
        state_dict[f"{prefix}.global_attention.key"] = weights[offset + 15].contiguous()
        state_dict[f"{prefix}.global_attention.value"] = weights[offset + 16].contiguous()
        _copy_layer_norm(state_dict, f"{prefix}.global_layer_norm1", weights[offset + 17], weights[offset + 18])
        state_dict[f"{prefix}.global_dense2.weight"] = weights[offset + 19].T.contiguous()
        state_dict[f"{prefix}.global_dense2.bias"] = weights[offset + 20].contiguous()
        _copy_layer_norm(state_dict, f"{prefix}.global_layer_norm2", weights[offset + 21], weights[offset + 22])

    return state_dict


def _convert_conv_branch(state_dict: dict[str, torch.Tensor], prefix: str, kernel: torch.Tensor, bias: torch.Tensor):
    state_dict[f"{prefix}.weight"] = kernel.permute(2, 1, 0).contiguous()
    state_dict[f"{prefix}.bias"] = bias.contiguous()


def _copy_layer_norm(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    weight: torch.Tensor,
    bias: torch.Tensor,
):
    state_dict[f"{prefix}.weight"] = weight.contiguous()
    state_dict[f"{prefix}.bias"] = bias.contiguous()


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    default_variant: str | None = "proteinbert"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
