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
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import DeepSeaConfig as Config
from multimolecule.models import DeepSeaForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream DeepSEA (Zhou & Troyanskaya, Nat. Methods 2015) one-hot encodes DNA as
# ["A", "G", "C", "T"]. The community PyTorch port distributed by Kipoi
# (https://zenodo.org/records/1466993/files/deepsea_predict.pth, CC-BY 3.0) was produced from the
# original Lua-Torch checkpoint via https://github.com/clcarwin/convert_torch_to_pytorch and keeps
# the upstream channel order. The first-layer input channels are permuted to MultiMolecule's
# ``[A, C, G, T]`` order in ``convert_checkpoint``.
ORIGINAL_VOCAB_LIST = ["A", "G", "C", "T"]

# The upstream module is the flat ``Sequential`` produced by ``convert_torch_to_pytorch``:
#   Sequential(
#     (2): Sequential(
#       (0)  SpatialConvolution      -> (320, 4, 1, 8)
#       (1)  ReLU
#       (2)  SpatialMaxPooling kW=4
#       (3)  Dropout(p=0.2)
#       (4)  SpatialConvolution      -> (480, 320, 1, 8)
#       (5)  ReLU
#       (6)  SpatialMaxPooling kW=4
#       (7)  Dropout(p=0.2)
#       (8)  SpatialConvolution      -> (960, 480, 1, 8)
#       (9)  ReLU
#       (10) Dropout(p=0.5)
#       (11) Reshape
#       (12) Sequential( Lambda, Linear(50880, 925) )
#       (13) ReLU
#       (14) Sequential( Lambda, Linear(925, 919) )
#       (15) Sigmoid
#     )
#   )
# Each parameterised module maps 1:1 onto the MultiMolecule module tree below.
# The target model is ``DeepSeaForSequencePrediction``; its backbone lives under ``model.`` and
# the 919-way chromatin-feature classifier is the ``sequence_head.decoder`` Linear.
CONV_LAYER_KEYS = [
    "model.encoder.layers.0",
    "model.encoder.layers.1",
    "model.encoder.layers.2",
]
ORIGINAL_CONV_KEYS = ["2.0", "2.4", "2.8"]
FC_LAYER_KEYS = ["model.encoder.fc_layers.0"]
ORIGINAL_FC_KEYS = ["2.12.1"]
DECODER_KEY = "sequence_head.decoder"
ORIGINAL_DECODER_KEY = "2.14.1"


def convert_checkpoint(convert_config):
    print(f"Converting DeepSEA checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    original_state_dict = _load_original_state_dict(convert_config.checkpoint_path)
    state_dict = _convert_checkpoint(original_state_dict)

    first_conv_key = f"{CONV_LAYER_KEYS[0]}.conv.weight"
    state_dict[first_conv_key] = convert_one_hot_embeddings(
        state_dict[first_conv_key],
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _load_original_state_dict(checkpoint_path: str) -> OrderedDict[str, torch.Tensor]:
    """Load the community PyTorch port of the upstream DeepSEA Lua-Torch checkpoint.

    The artifact is ``deepsea_predict.pth`` distributed via Kipoi at
    https://zenodo.org/records/1466993/files/deepsea_predict.pth (CC-BY 3.0). Pass the file via
    ``--checkpoint_path``.
    """
    if not (checkpoint_path and os.path.isfile(checkpoint_path)):
        raise FileNotFoundError(
            "Upstream DeepSEA checkpoint not found. Download `deepsea_predict.pth` from "
            "https://zenodo.org/records/1466993/files/deepsea_predict.pth and pass it via "
            "`--checkpoint_path`."
        )
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _convert_checkpoint(
    original_state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for new_prefix, old_prefix in zip(CONV_LAYER_KEYS, ORIGINAL_CONV_KEYS):
        # The community port stores convolutions as Conv2d with a unit `H` dimension; squeeze it
        # so the weight matches the MultiMolecule Conv1d (out, in, kW) layout.
        weight = original_state_dict[f"{old_prefix}.weight"]
        if weight.dim() == 4:
            if weight.size(2) != 1:
                raise ValueError(f"Unexpected conv weight shape {tuple(weight.shape)} for {old_prefix}; expected H=1.")
            weight = weight.squeeze(2)
        state_dict[f"{new_prefix}.conv.weight"] = weight.clone()
        state_dict[f"{new_prefix}.conv.bias"] = original_state_dict[f"{old_prefix}.bias"].clone()

    for new_prefix, old_prefix in zip(FC_LAYER_KEYS, ORIGINAL_FC_KEYS):
        state_dict[f"{new_prefix}.dense.weight"] = original_state_dict[f"{old_prefix}.weight"].clone()
        state_dict[f"{new_prefix}.dense.bias"] = original_state_dict[f"{old_prefix}.bias"].clone()

    state_dict[f"{DECODER_KEY}.weight"] = original_state_dict[f"{ORIGINAL_DECODER_KEY}.weight"].clone()
    state_dict[f"{DECODER_KEY}.bias"] = original_state_dict[f"{ORIGINAL_DECODER_KEY}.bias"].clone()

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
