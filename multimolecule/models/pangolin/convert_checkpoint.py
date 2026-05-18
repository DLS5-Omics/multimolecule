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

from multimolecule.models import PangolinConfig as Config
from multimolecule.models import PangolinModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream Pangolin one-hot encoding (`IN_MAP` in `pangolin/pangolin.py`) is ordered as ["A", "C", "G", "T"].
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# The canonical Pangolin v2 inference path averages three replicate networks for each tissue-specific model group.
# Groups 0, 2, 4, and 6 correspond to heart, liver, brain, and testis splice-site strength, respectively.
DEFAULT_ENSEMBLE_MEMBERS: list[list[str]] = [
    ["final.1.0.3.v2", "final.2.0.3.v2", "final.3.0.3.v2"],
    ["final.1.2.3.v2", "final.2.2.3.v2", "final.3.2.3.v2"],
    ["final.1.4.3.v2", "final.2.4.3.v2", "final.3.4.3.v2"],
    ["final.1.6.3.v2", "final.2.6.3.v2", "final.3.6.3.v2"],
]


def convert_checkpoint(convert_config):
    print(f"Converting Pangolin checkpoint at {convert_config.checkpoint_path}")
    members = convert_config.ensemble_members or DEFAULT_ENSEMBLE_MEMBERS
    if members and isinstance(members[0], str):
        members = [members]
    config = Config(num_tissues=len(members), num_ensemble=len(members[0]))

    model = Model(config)

    root = convert_config.checkpoint_path

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for tissue_index, tissue_members in enumerate(members):
        if len(tissue_members) != config.num_ensemble:
            raise ValueError("Every Pangolin tissue group must contain the same number of ensemble members.")
        for member_index, member in enumerate(tissue_members):
            member_path = os.path.join(root, member)
            if not os.path.exists(member_path):
                raise FileNotFoundError(
                    "Upstream Pangolin weights not found: "
                    f"{member_path}. Download the official tkzeng/Pangolin repository and pass "
                    "--checkpoint_path pointing at its `pangolin/models` directory."
                )
            member_state = _convert_checkpoint(member_path)
            member_state["projection.weight"] = convert_one_hot_embeddings(
                member_state["projection.weight"],
                old_vocab=ORIGINAL_VOCAB_LIST,
                new_vocab=new_vocab_list,
                convert_word_embeddings=convert_word_embeddings,
            )
            member_module_state = model.members[tissue_index][member_index].state_dict()
            for key, value in member_module_state.items():
                if key.endswith("num_batches_tracked") and key not in member_state:
                    member_state[key] = value
            for key, value in member_state.items():
                state_dict[f"members.{tissue_index}.{member_index}.{key}"] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(file: str) -> OrderedDict:
    original = torch.load(file, map_location="cpu", weights_only=True)
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in original.items():
        new_name = convert_original_state_dict_key(name)
        if new_name is None:
            continue
        state_dict[new_name] = value
    return state_dict


# Upstream `conv_lastN` heads: odd N -> 2-channel splice-site score, even N -> 1-channel usage score.
# conv_last1/2 -> tissue 0, conv_last3/4 -> tissue 1, conv_last5/6 -> tissue 2, conv_last7/8 -> tissue 3.
def _convert_conv_last(index: int, suffix: str) -> str:
    tissue = (index - 1) // 2
    kind = "score" if index % 2 == 1 else "usage"
    return f"prediction.{kind}.{tissue}.{suffix}"


name_mapping = {
    "bn1": "norm1",
    "bn2": "norm2",
}


def convert_original_state_dict_key(name: str) -> str | None:
    if name.startswith("conv1."):
        return name.replace("conv1.", "projection.", 1)
    if name.startswith("skip."):
        return name.replace("skip.", "encoder.conv.", 1)
    if name.startswith("resblocks."):
        _, idx, rest = name.split(".", 2)
        block_index = int(idx)
        stage_index, block_in_stage = divmod(block_index, 4)
        for old, new in name_mapping.items():
            rest = rest.replace(old, new)
        return f"encoder.stages.{stage_index}.blocks.{block_in_stage}.{rest}"
    if name.startswith("convs."):
        _, idx, rest = name.split(".", 2)
        return f"encoder.stages.{int(idx)}.conv.{rest}"
    if name.startswith("conv_last"):
        prefix, suffix = name.split(".", 1)
        return _convert_conv_last(int(prefix.replace("conv_last", "")), suffix)
    raise ValueError(f"Unexpected upstream key: {name}")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    ensemble_members: list[str] | list[list[str]] | None = None


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
