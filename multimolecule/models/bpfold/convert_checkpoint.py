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
from pathlib import Path

import chanfig
import torch

from multimolecule.models import BpfoldConfig as Config
from multimolecule.models import BpfoldModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config


def convert_checkpoint(convert_config) -> None:
    print(f"Converting BPfold checkpoints at {convert_config.checkpoint_path}")
    checkpoint_dir = Path(convert_config.checkpoint_path)
    checkpoint_paths = sorted(checkpoint_dir.glob("BPfold_*-6.pth"))
    if len(checkpoint_paths) != 6:
        checkpoint_paths = sorted(checkpoint_dir.glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No BPfold checkpoints found in {checkpoint_dir}.")

    config = Config(num_members=len(checkpoint_paths))
    config.architectures = ["BpfoldModel"]
    model = Model(config)
    vocab_list = list(get_alphabet("streamline").vocabulary)

    state_dict = {}
    for member_index, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected a state dict checkpoint at {checkpoint_path}, but got {type(checkpoint)}.")
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            key = _convert_original_state_dict_key(key)
            if key == "embeddings.weight":
                (value,) = convert_word_embeddings(
                    value,
                    old_vocab=original_vocab_list,
                    new_vocab=vocab_list,
                    seed=1016,
                )
                value[vocab_list.index("N")] = value[vocab_list.index("U")]
            state_dict[f"members.{member_index}.{key}"] = value

    energy_path = Path(convert_config.energy_path)
    if not energy_path.exists():
        raise FileNotFoundError(f"BPfold energy table not found: {energy_path}")
    state_dict.update(_bpfold_energy_tables_from_file(str(energy_path)))
    state_dict["criterion.pos_weight"] = model.criterion.pos_weight

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = get_alphabet("streamline")
    tokenizer_config["unk_token"] = "N"

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_original_state_dict_key(key: str) -> str:
    replacements = {
        "transformer.conv_layers.": "encoder.pairwise_blocks.",
        "transformer.layers.": "encoder.layer.",
        ".in_norm.": ".layer_norm.",
        ".mhsa.": ".attention.",
        ".ffn.0.": ".intermediate.layer_norm.",
        ".ffn.1.": ".intermediate.dense.",
        ".ffn.4.": ".output.dense.",
        ".dynpos.": ".dynamic_position_bias.",
        ".alibi.": ".alibi_position_bias.",
        ".weights": ".query_key_value_weight",
        ".out_w": ".output_weight",
        ".out_bias": ".output_bias",
        ".in_bias": ".query_key_value_bias",
    }
    if key.startswith("emb."):
        key = f"embeddings.{key.removeprefix('emb.')}"
    for source, target in replacements.items():
        key = key.replace(source, target)
    key = re.sub(r"(encoder\.pairwise_blocks\.\d+)\.conv\.0\.", r"\1.conv.", key)
    key = re.sub(r"(encoder\.pairwise_blocks\.\d+)\.conv\.1\.", r"\1.batch_norm.", key)
    key = re.sub(r"(encoder\.pairwise_blocks\.\d+)\.conv\.2\.excitation\.0\.", r"\1.squeeze_excitation.dense1.", key)
    key = re.sub(r"(encoder\.pairwise_blocks\.\d+)\.conv\.2\.excitation\.2\.", r"\1.squeeze_excitation.dense2.", key)
    key = re.sub(r"(encoder\.pairwise_blocks\.\d+)\.res\.0\.", r"\1.residual.", key)
    return key


def _bpfold_energy_tables_from_file(path: str) -> dict[str, torch.Tensor]:
    vocab_list = get_alphabet("nucleobase", prepend_tokens=[]).vocabulary
    canonical_pairs = ("GC", "CG", "AU", "UA", "GU", "UG")
    pair_to_index = {pair: index for index, pair in enumerate(canonical_pairs)}
    energy_table: dict[str, float] = {}
    min_by_suffix: dict[str, float] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            motif, energy_string = line.split()
            energy = float(energy_string)
            energy_table[motif] = energy
            suffix = motif[motif.find("_") :]
            min_by_suffix[suffix] = min(min_by_suffix.get(suffix, energy), energy)

    outer_shape, inner_chain_shape, inner_hairpin_shape = _energy_table_shapes(
        num_bases=len(vocab_list),
        num_pair_types=len(canonical_pairs),
        motif_radius=3,
    )
    outer = torch.zeros(outer_shape)
    inner_chain = torch.zeros(inner_chain_shape)
    inner_hairpin = torch.zeros(inner_hairpin_shape)
    for motif, energy in energy_table.items():
        suffix = motif[motif.find("_") :]
        normalized = energy / min_by_suffix[suffix]
        sequence = motif[: motif.find("_")]
        pair_index = pair_to_index.get(sequence[0] + sequence[-1])
        if pair_index is None:
            continue
        if "-" not in suffix:
            distance = int(suffix.rsplit("_", 1)[1])
            code = _encode_sequence(sequence[1:-1], vocab_list)
            if distance < inner_hairpin.size(1):
                inner_hairpin[pair_index, distance, code] = normalized
        elif suffix == "_0_7-3":
            code = _encode_sequence(sequence[1:-1], vocab_list)
            inner_chain[pair_index, code] = normalized
        else:
            distance_string, chain_break_string = suffix.rsplit("-", 1)
            distance = int(distance_string.rsplit("_", 1)[1])
            left_length = int(chain_break_string)
            right_length = distance - left_length - 1
            middle = sequence[1:-1]
            left_code = _encode_sequence(middle[:left_length], vocab_list)
            right_code = _encode_sequence(middle[left_length:], vocab_list)
            outer[pair_index, left_length, right_length, left_code, right_code] = normalized

    return {
        "outer_energy": outer,
        "inner_chain_energy": inner_chain,
        "inner_hairpin_energy": inner_hairpin,
    }


def _encode_sequence(sequence: str, vocab_list: tuple[str, ...]) -> int:
    code = 0
    for base in sequence:
        code = code * len(vocab_list) + vocab_list.index(base)
    return code


def _energy_table_shapes(
    num_bases: int,
    num_pair_types: int,
    motif_radius: int,
) -> tuple[tuple[int, int, int, int, int], tuple[int, int], tuple[int, int, int]]:
    motif_code_size = num_bases**motif_radius
    chain_code_size = num_bases ** (2 * motif_radius)
    max_hairpin_distance = 2 * motif_radius
    return (
        (num_pair_types, motif_radius + 1, motif_radius + 1, motif_code_size, motif_code_size),
        (num_pair_types, chain_code_size),
        (num_pair_types, max_hairpin_distance + 1, chain_code_size),
    )


original_vocab_list = ["A", "U", "C", "G", "<cls>", "<eos>", "<pad>"]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    energy_path: str = os.path.join(root, "key.energy")


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
