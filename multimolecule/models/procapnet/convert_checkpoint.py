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

import glob
import os
from collections import OrderedDict
from pathlib import Path

import chanfig
import torch

from multimolecule.models import ProCapNetConfig as Config
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.models.procapnet.modeling_procapnet import (
    ProCapNetForProfilePrediction,
)
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream ProCapNet (Cochran et al. 2024, kundajelab/ProCapNet, bpnet-lite) one-hot encodes DNA as
# ["A", "C", "G", "T"] (see src/2_train_models/data_loading.py).
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]

# Upstream (PyTorch / bpnet-lite) state-dict key -> MultiMolecule key prefix mapping.
# ProCapNet is PyTorch native, so layouts match torch Conv1d / Linear directly; only the
# stem input channels need reordering to MultiMolecule nucleotide order.
_NAME_MAP = {
    "iconv": "model.encoder.stem.conv",
    "fconv": "profile_count_head.profile",
    "linear": "profile_count_head.count",
}


def _convert_name(key: str, config: Config) -> str:
    if key.startswith("rconvs."):
        idx = int(key.split(".")[1])
        rest = key.split(".", 2)[2]
        return f"model.encoder.layer.{idx}.conv.{rest}"
    head, rest = key.split(".", 1)
    if head not in _NAME_MAP:
        raise ValueError(f"Unexpected upstream ProCapNet state-dict key: {key}")
    return f"{_NAME_MAP[head]}.{rest}"


def convert_checkpoint(convert_config):
    print(f"Converting ProCapNet checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = ProCapNetForProfilePrediction(config)

    root = convert_config.checkpoint_path
    if root is None:
        raise FileNotFoundError("ProCapNet conversion requires a checkpoint file or directory, got None.")
    root_path = Path(root)
    if root_path.is_file():
        ckpt = str(root_path)
    else:
        candidates = sorted(glob.glob(os.path.join(str(root_path), "**", "*.state_dict.torch"), recursive=True))
        if not candidates:
            raise FileNotFoundError(f"No '*.state_dict.torch' checkpoint found under {root_path}")
        ckpt = candidates[0]
        print(f"Using checkpoint {ckpt}")

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    tokenizer_config["model_max_length"] = config.sequence_length

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(ckpt, config)
    key = "model.encoder.stem.conv.weight"
    if key not in state_dict:
        raise KeyError(f"Expected stem convolution weight {key!r} is missing; cannot convert input channels.")
    state_dict[key] = convert_one_hot_embeddings(
        state_dict[key],
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    reference = model.state_dict()
    for key, value in reference.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(file: str, config: Config) -> OrderedDict:
    # ProCapNet is PyTorch native (bpnet-lite); the checkpoint is a clean ``state_dict`` ``OrderedDict``
    # with torch Conv1d / Linear layouts, so only the key names are translated.
    original = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(original, dict):
        raise TypeError(f"Expected a state_dict OrderedDict, got {type(original)!r}")

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in original.items():
        state_dict[_convert_name(key, config)] = value.clone()
    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
