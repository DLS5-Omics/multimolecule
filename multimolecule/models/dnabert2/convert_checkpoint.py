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

import json
import os
import re
import tempfile

import chanfig
import torch

from multimolecule.models import DnaBert2Config as Config
from multimolecule.models import DnaBert2ForMaskedLM as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint

torch.manual_seed(1016)


def convert_checkpoint(convert_config):
    print(f"Converting DNABERT-2 checkpoint at {convert_config.checkpoint_path}")
    config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config.hidden_dropout = config.pop("hidden_dropout_prob", 0.1)
    config.attention_dropout = config.pop("attention_probs_dropout_prob", 0.0)
    config.position_embedding_type = "alibi"
    # Remove fields from the original config that are not part of DnaBert2Config
    for field in ("auto_map", "classifier_dropout", "gradient_checkpointing", "torch_dtype", "dtype"):
        config.pop(field, None)
    config = Config.from_dict(config)
    del config._name_or_path
    config.architectures = ["DnaBert2ForMaskedLM"]

    model = Model(config)

    ckpt = torch.load(
        os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
    )
    state_dict = _convert_checkpoint(config, ckpt)

    load_checkpoint(model, state_dict)
    tokenizer_config = _get_tokenizer_config(convert_config.checkpoint_path)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict):
    state_dict = {}
    hidden_size = config.hidden_size

    for key, value in original_state_dict.items():
        # Handle Wqkv split into separate Q, K, V
        match = re.match(r"bert\.encoder\.layer\.(\d+)\.attention\.self\.Wqkv\.(weight|bias)", key)
        if match:
            layer_idx = match.group(1)
            param_type = match.group(2)
            prefix = f"model.encoder.layer.{layer_idx}.attention.self"
            if param_type == "weight":
                # value shape: [2304, 768] -> split into 3 x [768, 768]
                q, k, v = value.split(hidden_size, dim=0)
                state_dict[f"{prefix}.query.weight"] = q
                state_dict[f"{prefix}.key.weight"] = k
                state_dict[f"{prefix}.value.weight"] = v
            else:
                # value shape: [2304] -> split into 3 x [768]
                q, k, v = value.split(hidden_size, dim=0)
                state_dict[f"{prefix}.query.bias"] = q
                state_dict[f"{prefix}.key.bias"] = k
                state_dict[f"{prefix}.value.bias"] = v
            continue

        # Rename LayerNorm -> layer_norm, layernorm -> layer_norm
        new_key = key.replace("LayerNorm", "layer_norm")
        new_key = new_key.replace("layernorm", "layer_norm")

        # Rename MLP layers: gated_layers -> up_proj, wo -> down_proj
        new_key = new_key.replace(".mlp.gated_layers.", ".mlp.up_proj.")
        new_key = new_key.replace(".mlp.wo.", ".mlp.down_proj.")

        # Rename bert.* -> model.*
        if new_key.startswith("bert"):
            new_key = "model" + new_key[4:]
            state_dict[new_key] = value
            continue

        # Rename cls.predictions.* -> lm_head.*
        if new_key.startswith("cls"):
            new_key = "lm_head" + new_key[15:]
            state_dict[new_key] = value
            # decoder.bias should also be mapped to lm_head.bias
            if new_key == "lm_head.decoder.bias":
                state_dict["lm_head.bias"] = value
            continue

        state_dict[new_key] = value

    return state_dict


def _remap_special_token(token: str) -> str:
    """Remap BERT-style special tokens to multimolecule convention.

    ``[UNK]`` -> ``<unk>``, ``[CLS]`` -> ``<cls>``, ``[SEP]`` -> ``<eos>``, etc.
    """
    if token.startswith("[") and token.endswith("]"):
        token = token.replace("[", "<").replace("]", ">").lower()
    if token == "<sep>":
        token = "<eos>"
    return token


def _get_tokenizer_config(checkpoint_path):
    """Build a tokenizer_config dict compatible with the shared save_checkpoint.

    Converts the BPE tokenizer's special tokens from BERT convention ([CLS], [SEP], etc.)
    to multimolecule convention (<cls>, <eos>, etc.) and returns a config dict that
    PreTrainedTokenizerFast can be constructed from.
    """
    tokenizer_json_src = os.path.join(checkpoint_path, "tokenizer.json")
    with open(tokenizer_json_src) as f:
        tokenizer_data = json.load(f)

    # Remap added_tokens
    for token in tokenizer_data.get("added_tokens", []):
        token["content"] = _remap_special_token(token["content"])

    # Remap model vocab keys
    vocab = tokenizer_data["model"]["vocab"]
    tokenizer_data["model"]["vocab"] = {_remap_special_token(k): v for k, v in vocab.items()}

    # Remap post_processor template tokens
    _remap_post_processor(tokenizer_data.get("post_processor", {}))

    # Write converted tokenizer.json to a temp file for PreTrainedTokenizerFast to load
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(tokenizer_data, tmp)
    tmp.close()

    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "tokenizer_file": tmp.name,
        "unk_token": "<unk>",
        "cls_token": "<cls>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }


def _remap_post_processor(post_processor):
    """Recursively remap special token strings in post_processor."""
    if isinstance(post_processor, dict):
        # Remap special_tokens keys
        if "special_tokens" in post_processor:
            new_st = {}
            for k, v in post_processor["special_tokens"].items():
                new_key = _remap_special_token(k)
                v["id"] = _remap_special_token(v["id"])
                v["tokens"] = [_remap_special_token(t) for t in v.get("tokens", [])]
                new_st[new_key] = v
            post_processor["special_tokens"] = new_st
        for key, value in post_processor.items():
            if key == "special_tokens":
                continue
            if key == "id" and isinstance(value, str):
                post_processor[key] = _remap_special_token(value)
            elif key == "tokens" and isinstance(value, list):
                post_processor[key] = [_remap_special_token(t) for t in value]
            else:
                _remap_post_processor(value)
    elif isinstance(post_processor, list):
        for item in post_processor:
            _remap_post_processor(item)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]

    def post(self):
        if self.output_path is None:
            self.output_path = self.checkpoint_path.split("/")[-1].lower().rsplit("-", 1)[0].replace("-", "")
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
