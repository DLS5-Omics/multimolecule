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

from multimolecule.models import SpTransformerConfig as Config
from multimolecule.models import SpTransformerModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream SpTransformer one-hot encoding (`IN_MAP` in `tasks_annotate_mutations.py`) is ordered as
# ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    if not convert_config.checkpoint_path:
        raise ValueError("checkpoint_path must point to the official SpTransformer_pytorch.ckpt checkpoint.")
    print(f"Converting SpTransformer checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    original = torch.load(convert_config.checkpoint_path, map_location="cpu", weights_only=False)
    original_state_dict = original["state_dict"] if "state_dict" in original else original

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in original_state_dict.items():
        new_name = convert_original_state_dict_key(name)
        if new_name is None:
            continue
        state_dict[new_name] = value

    # The first convolution of the trainable input path and each feature encoder consumes upstream one-hot channels.
    for key in list(state_dict.keys()):
        if key.endswith("conv1.weight") and state_dict[key].shape[1] == len(ORIGINAL_VOCAB_LIST):
            state_dict[key] = convert_one_hot_embeddings(
                state_dict[key],
                old_vocab=ORIGINAL_VOCAB_LIST,
                new_vocab=new_vocab_list,
                convert_word_embeddings=convert_word_embeddings,
            )

    target = model.state_dict()
    for key, value in target.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def convert_original_state_dict_key(name: str) -> str | None:
    # SpliceAI-style feature extractors: `encoder.<i>.*` -> `feature_encoders.<i>.*`. The upstream encoder
    # output projections (`splice_output`, `tissue_output`) are unused by the feature path and dropped.
    if name.startswith("encoder."):
        idx, rest = name.split(".", 2)[1], name.split(".", 2)[2]
        if rest.startswith("splice_output") or rest.startswith("tissue_output"):
            return None
        rest = rest.replace(".bn1.", ".norm1.").replace(".bn2.", ".norm2.")
        return f"feature_encoders.{idx}.{rest}"
    # Trainable input path: `conv1.0` / `conv1.1` -> projection.conv1 / projection.conv2.
    if name.startswith("conv1.0."):
        return name.replace("conv1.0.", "projection.conv1.", 1)
    if name.startswith("conv1.1."):
        return name.replace("conv1.1.", "projection.conv2.", 1)
    # Feature-fusion projection.
    if name.startswith("conv2."):
        return name.replace("conv2.", "projection.conv.", 1)
    # Axial positional embedding (newer axial-positional-embedding uses `weights.0` / `weights.1`).
    if name == "attn.pos_emb.weights_0" or name == "attn.pos_emb.weights.0":
        return "encoder.position_embeddings.weights.0"
    if name == "attn.pos_emb.weights_1" or name == "attn.pos_emb.weights.1":
        return "encoder.position_embeddings.weights.1"
    if name == "attn.norm.weight":
        return "encoder.layer_norm.weight"
    if name == "attn.norm.bias":
        return "encoder.layer_norm.bias"
    # Sinkhorn transformer attention layers.
    if name.startswith("attn.attn.layers.layers."):
        rest = name[len("attn.attn.layers.layers.") :]
        layer, sub, tail = rest.split(".", 2)
        if sub == "0":  # attention sublayer
            tail = tail.replace("norm.", "attention.layer_norm.", 1)
            tail = tail.replace("fn.to_q.", "attention.self.query.", 1)
            tail = tail.replace("fn.to_kv.", "attention.self.key_value.", 1)
            tail = tail.replace("fn.to_out.", "attention.output.dense.", 1)
        else:  # feed-forward sublayer
            tail = tail.replace("norm.", "intermediate.layer_norm.", 1)
            tail = tail.replace("fn.fn.w1.", "intermediate.dense.", 1)
            tail = tail.replace("fn.fn.w2.", "output.dense.", 1)
        return f"encoder.layer.{layer}.{tail}"
    # Original output heads.
    if name.startswith("splice.") or name.startswith("usage."):
        return f"prediction.{name}"
    raise ValueError(f"Unexpected upstream key: {name}")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
