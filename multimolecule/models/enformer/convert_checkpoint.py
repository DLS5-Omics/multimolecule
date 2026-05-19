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
import re
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import EnformerConfig as Config
from multimolecule.models import EnformerForTokenPrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Enformer one-hot encodes DNA in the order ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# The source PyTorch checkpoint exposes the same modules twice: once under the canonical
# ``stem``/``conv_tower``/``transformer``/``final_pointwise`` names and once under the ``_trunk.*``
# Sequential wrapper that references the very same parameters. We only translate the canonical keys;
# the ``_trunk.*`` aliases are dropped.


def convert_checkpoint(convert_config):
    print(f"Converting Enformer checkpoint at {convert_config.checkpoint_path}")
    config = Config(species=convert_config.species, sequence_length=196_608, use_precomputed_gamma_basis=True)
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    tokenizer_config["model_max_length"] = config.sequence_length

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, convert_config.species)
    gamma_position_basis = _load_precomputed_gamma_position_basis()
    for index in range(config.num_hidden_layers):
        state_dict[f"model.encoder.layers.{index}.attention.gamma_position_basis"] = gamma_position_basis.clone()
    if "model.encoder.stem.conv1.weight" not in state_dict:
        raise KeyError(
            "Expected key 'model.encoder.stem.conv1.weight' after conversion but it was not "
            "found. The reference Enformer checkpoint layout may have changed."
        )
    key = "model.encoder.stem.conv1.weight"
    state_dict[key] = convert_one_hot_embeddings(
        state_dict[key],
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    reference_state = model.state_dict()
    for key, value in reference_state.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_name(name: str, species: str) -> str | None:
    # Drop the ``_trunk.*`` Sequential aliases; they reference the same parameters as the
    # canonical names below.
    if name.startswith("_trunk."):
        return None

    # Output heads: keep only the requested species head and map it to the shared
    # TokenPredictionHead decoder. The task head lives on EnformerForTokenPrediction directly,
    # so it is not prefixed with the ``model.`` backbone prefix.
    head_match = re.match(r"^_heads\.(\w+)\.0\.(weight|bias)$", name)
    if head_match is not None:
        head_species, param = head_match.groups()
        if head_species != species:
            return None
        return f"token_head.decoder.{param}"

    # Everything else maps onto the EnformerModel backbone, which lives under the ``model.``
    # prefix on EnformerForTokenPrediction.
    backbone: str | None = None

    # Stem.
    if name.startswith("stem.0."):
        backbone = name.replace("stem.0.", "encoder.stem.conv1.")
    elif name.startswith("stem.1.fn.0."):
        backbone = name.replace("stem.1.fn.0.", "encoder.stem.conv_block.batch_norm1.")
    elif name.startswith("stem.1.fn.2."):
        backbone = name.replace("stem.1.fn.2.", "encoder.stem.conv_block.conv1.")
    elif name.startswith("stem.2."):
        backbone = name.replace("stem.2.", "encoder.stem.pool.")

    # Conv tower stages.
    conv_match = re.match(r"^conv_tower\.(\d+)\.(\d+)\.(.+)$", name)
    if backbone is None and conv_match is not None:
        idx, sub, rest = conv_match.groups()
        if sub == "0":
            rest = rest.replace("0.", "batch_norm1.", 1) if rest.startswith("0.") else rest
            rest = rest.replace("2.", "conv1.", 1) if rest.startswith("2.") else rest
            backbone = f"encoder.conv_tower.{idx}.conv_block.{rest}"
        elif sub == "1":
            rest = rest.replace("fn.0.", "batch_norm1.", 1) if rest.startswith("fn.0.") else rest
            rest = rest.replace("fn.2.", "conv1.", 1) if rest.startswith("fn.2.") else rest
            backbone = f"encoder.conv_tower.{idx}.conv_block_residual.{rest}"
        elif sub == "2":
            backbone = f"encoder.conv_tower.{idx}.pool.{rest}"

    # Transformer blocks.
    tr_match = re.match(r"^transformer\.(\d+)\.(\d+)\.fn\.(.+)$", name)
    if backbone is None and tr_match is not None:
        idx, sub, rest = tr_match.groups()
        if sub == "0":
            if rest.startswith("0."):
                backbone = f"encoder.layers.{idx}.attention.layer_norm.{rest[2:]}"
            elif rest.startswith("1."):
                backbone = f"encoder.layers.{idx}.attention.{rest[2:]}"
        elif sub == "1":
            if rest.startswith("0."):
                backbone = f"encoder.layers.{idx}.intermediate.layer_norm.{rest[2:]}"
            elif rest.startswith("1."):
                backbone = f"encoder.layers.{idx}.intermediate.dense1.{rest[2:]}"
            elif rest.startswith("4."):
                backbone = f"encoder.layers.{idx}.intermediate.dense2.{rest[2:]}"

    # Final pointwise head.
    if backbone is None and name.startswith("final_pointwise.1.0."):
        backbone = name.replace("final_pointwise.1.0.", "encoder.head.conv_block.batch_norm1.")
    elif backbone is None and name.startswith("final_pointwise.1.2."):
        backbone = name.replace("final_pointwise.1.2.", "encoder.head.conv_block.conv1.")

    if backbone is None:
        raise KeyError(f"Unhandled reference checkpoint key: {name}")
    return f"model.{backbone}"


def _convert_checkpoint(file, species: str) -> OrderedDict:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    if not file or not os.path.exists(file):
        raise FileNotFoundError(f"Enformer checkpoint not found: {file!r}")

    original = torch.load(file, map_location="cpu", weights_only=True)
    for name, value in original.items():
        new_name = _convert_name(name, species)
        if new_name is None:
            continue
        state_dict[new_name] = value
    return state_dict


def _load_precomputed_gamma_position_basis() -> torch.Tensor:
    try:
        from enformer_pytorch.modeling_enformer import TF_GAMMAS
    except ImportError as error:
        raise ImportError(
            "Converting the official Enformer checkpoint requires enformer_pytorch so the "
            "released fixed gamma positional basis can be embedded in the checkpoint."
        ) from error
    return TF_GAMMAS.detach().to(torch.float32).contiguous()


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    species: str = "human"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
