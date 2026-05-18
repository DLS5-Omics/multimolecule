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

from multimolecule.models import MaxEntScanConfig as Config
from multimolecule.models import MaxEntScanModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import save_checkpoint
from multimolecule.models.maxentscan.modeling_maxentscan import SCORE3_SUBMODEL_POSITIONS
from multimolecule.tokenisers.dna.utils import get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# MaxEntScan is a maximum-entropy model with NO trainable weights and NO upstream torch checkpoint.
# The "parameters" are the fixed maximum-entropy probability tables published with the original
# Yeo & Burge (2004) tool (http://genes.mit.edu/burgelab/maxent/download/):
#   - score5: the 16384-entry `me2x5` maximum-entropy probabilities over the 7 non-consensus
#             positions of the 9-mer (the GT consensus background ratios are fixed constants
#             baked into the model as buffers).
#   - score3: the nine maximum-entropy decomposition probability matrices `me2x3acc1..9`
#             (the AG consensus background ratios are fixed constants baked into the model).
# The original plain-text tables are bundled here, verbatim and in their native one-float-per-line
# order (which equals base-4 / the published `splice5sequences` enumeration), as:
#   - `score5_me2x5.txt`     : the 16384 `me2x5` probabilities.
#   - `score3_me2x3acc.txt`  : `me2x3acc1..9` concatenated, with a header recording each size.
# These values are bundled into the score-table buffers registered by `MaxEntScanScorer` (the model
# also self-loads them in `__init__`, so this converter mainly validates the bundled tables and emits
# a standard, reloadable checkpoint).

SCORE5_TABLE = "score5_me2x5.txt"
SCORE3_TABLE = "score3_me2x3acc.txt"


def _parse_probability_file(path: str, expected_size: int) -> torch.Tensor:
    """Parse a MaxEntScan plain-text probability table (one float per line) into a 1-D tensor."""
    with open(path) as file:
        values = [float(line) for line in file if line.strip() and not line.lstrip().startswith("#")]
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() != expected_size:
        raise ValueError(f"{path}: expected {expected_size} probabilities, found {tensor.numel()}")
    return tensor


def _parse_score3_tables(path: str, sizes: list[int]) -> list[torch.Tensor]:
    """Parse the concatenated `me2x3acc1..9` plain-text tables back into nine 1-D tensors."""
    with open(path) as file:
        values = [float(line) for line in file if line.strip() and not line.lstrip().startswith("#")]
    if len(values) != sum(sizes):
        raise ValueError(f"{path}: expected {sum(sizes)} probabilities, found {len(values)}")
    tables, offset = [], 0
    for size in sizes:
        tables.append(torch.tensor(values[offset : offset + size], dtype=torch.float32))
        offset += size
    return tables


def _load_published_tables(root: str, config: Config) -> OrderedDict[str, torch.Tensor] | None:
    """
    Build the score-table buffers from the bundled published MaxEntScan maximum-entropy matrices.

    Files are read from ``root`` (defaulting to this package directory) so the converter can be
    pointed at a freshly downloaded upstream release. Returns ``None`` when the table files are
    not present.
    """
    base = root if root else os.path.dirname(__file__)
    if config.mode == "score5":
        path = os.path.join(base, SCORE5_TABLE)
        if not os.path.exists(path):
            return None
        return OrderedDict({"scorer.score5_me2x5": _parse_probability_file(path, 4**7)})
    path = os.path.join(base, SCORE3_TABLE)
    if not os.path.exists(path):
        return None
    sizes = [4 ** len(positions) for positions in SCORE3_SUBMODEL_POSITIONS]
    tables = _parse_score3_tables(path, sizes)
    return OrderedDict({f"scorer.score3_table_{i}": table for i, table in enumerate(tables)})


def convert_checkpoint(convert_config):
    config = Config(mode=convert_config.mode)
    print(f"Converting MaxEntScan tables for mode={config.mode}")
    model = Model(config)

    tables = _load_published_tables(convert_config.checkpoint_path, config)
    if tables is None:
        raise FileNotFoundError(
            "Published MaxEntScan tables not found. Expected the bundled "
            f"{SCORE5_TABLE!r}/{SCORE3_TABLE!r} under the checkpoint path."
        )
    # Assign the published tables into the score-table buffers in place (the model already
    # self-loads the same values in `__init__`; this re-applies them from `checkpoint_path`).
    for name, value in tables.items():
        module = model
        *path, attr = name.split(".")
        for part in path:
            module = getattr(module, part)
        getattr(module, attr).copy_(value)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    tokenizer_config["model_max_length"] = config.window

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


class ConvertConfig(ConvertConfig_):
    checkpoint_path: str = os.path.dirname(__file__)
    mode: str = "score5"
    root: str = os.path.dirname(__file__)
    output_path: str = "maxentscan-score5"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
