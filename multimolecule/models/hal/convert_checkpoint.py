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

import hashlib
import os
import urllib.request
from collections import OrderedDict
from itertools import product

import chanfig
import numpy as np
import torch

from multimolecule.models import HalConfig as Config
from multimolecule.models import HalModel as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# The model index map in `modeling_hal.HalEmbedding` uses the streamline RNA order ["A", "C", "G", "U"].
# The published HAL artifact enumerates rows in the author's order ["A", "T", "C", "G"], so conversion maps
# each hexamer string into the model's base-4 feature index explicitly, with upstream T mapped to RNA U.
ORIGINAL_NUCLEOTIDES = ["A", "C", "G", "U"]

# Original published HAL hexamer-coefficient artifact from Rosenberg et al. 2015 Cell
# ("Learning the Sequence Determinants of Alternative Splicing from Millions of Random Sequences").
# The coefficients are computed by the authors' `Cell2015_N7_A5SS_Model.ipynb` notebook in the
# official repository https://github.com/Alex-Rosenberg/cell-2015 and are archived by
# Alexander B. Rosenberg at the canonical Zenodo deposit below.
HAL_COEFFICIENTS_URL = "https://zenodo.org/record/1466088/files/HAL_mer_scores.npz?download=1"
HAL_COEFFICIENTS_MD5 = "4db740f6b72345db5303c106ed6aad61"

# The published artifact stores a (4096, 8) matrix: hexamer rows in the author's enumeration order
# over bases ["A", "T", "C", "G"], and eight position-specific regions of the A5SS splicing model.
# The HAL formula averages those eight regions into one position-agnostic hexamer effect used with
# normalized hexamer-frequency features.
HAL_BASES = ["A", "T", "C", "G"]


def hexamer_to_index(hexamer: str, nucleobase_size: int) -> int:
    r"""Map a hexamer string (DNA or RNA, T treated as U) to its base-`nucleobase_size` feature index."""
    index = 0
    for base in hexamer:
        base = "U" if base in ("T", "t") else base.upper()
        index = index * nucleobase_size + ORIGINAL_NUCLEOTIDES.index(base)
    return index


def load_hexamer_coefficients(path: str, config: Config) -> dict[str, float]:
    r"""
    Load a user-provided HAL hexamer coefficient table.

    The expected format is a two-column whitespace- or comma-separated table::

        AAAAAA   0.0123
        AAAAAC  -0.0456
        ...

    Returns a mapping from hexamer string to coefficient.
    """
    coefficients: dict[str, float] = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            hexamer, value = parts[0], parts[1]
            if len(hexamer) != config.kmer_size:
                continue
            coefficients[hexamer] = float(value)
    if len(coefficients) != config.num_kmers:
        raise ValueError(f"Expected {config.num_kmers} hexamer coefficients, found {len(coefficients)}")
    return coefficients


def make_hexamer_list(kmer_size: int) -> list[str]:
    r"""Reproduce the author's hexamer enumeration order over the bases ``["A", "T", "C", "G"]``."""
    mers = list(HAL_BASES)
    for _ in range(kmer_size - 1):
        mers = [prefix + base for prefix in mers for base in HAL_BASES]
    return mers


def fetch_hal_coefficients(config: Config, cache_path: str | None = None) -> dict[str, float]:
    r"""
    Fetch and parse the original published HAL hexamer coefficient artifact.

    Downloads the Rosenberg et al. 2015 ``HAL_mer_scores.npz`` archive, verifies it against the
    published MD5, and returns a mapping from hexamer string to its averaged effect.
    """
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as file:
            raw = file.read()
    else:
        print(f"Downloading published HAL coefficients from {HAL_COEFFICIENTS_URL}")
        with urllib.request.urlopen(HAL_COEFFICIENTS_URL) as response:  # noqa: S310
            raw = response.read()
        if cache_path:
            with open(cache_path, "wb") as file:
                file.write(raw)

    digest = hashlib.md5(raw).hexdigest()
    if digest != HAL_COEFFICIENTS_MD5:
        raise ValueError(
            f"Downloaded HAL coefficient artifact MD5 {digest} does not match the published "
            f"value {HAL_COEFFICIENTS_MD5}; refusing to convert untrusted weights."
        )

    import io

    with np.load(io.BytesIO(raw)) as data:
        weights = np.asarray(data["weights"], dtype=np.float64)
    if weights.shape != (config.num_kmers, config.num_regions):
        raise ValueError(
            f"Expected HAL coefficients with shape {(config.num_kmers, config.num_regions)}, got {weights.shape}"
        )
    effect = weights.mean(axis=1)
    hexamers = make_hexamer_list(config.kmer_size)
    return {hexamer: float(value) for hexamer, value in zip(hexamers, effect)}


def convert_checkpoint(convert_config):
    print(f"Converting HAL checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"
    tokenizer_config["model_max_length"] = config.region_length

    weight = torch.zeros(config.num_labels, config.num_features)

    checkpoint_path = convert_config.checkpoint_path
    if checkpoint_path and os.path.isfile(checkpoint_path) and not checkpoint_path.endswith(".npz"):
        # A user-provided two-column hexamer/coefficient text table.
        coefficients = load_hexamer_coefficients(checkpoint_path, config)
    else:
        # The published HAL hexamer coefficients from Rosenberg et al. 2015 (Cell). The original
        # artifact is fetched and verified against its published MD5; ``--checkpoint_path`` may
        # point at a locally cached copy of the ``HAL_mer_scores.npz`` archive.
        cache_path = checkpoint_path if checkpoint_path.endswith(".npz") else None
        coefficients = fetch_hal_coefficients(config, cache_path=cache_path)

    for hexamer, value in coefficients.items():
        index = hexamer_to_index(hexamer, config.nucleobase_size)
        weight[0, index] = value

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    state_dict["prediction.prediction.weight"] = weight
    # The k-mer index map is a deterministic constant rebuilt from config in `HalEmbedding.forward`;
    # it is intentionally not added to the checkpoint (see implementation_guide.md, Buffers).

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def enumerate_hexamers(kmer_size: int) -> list[str]:
    r"""Enumerate all hexamers in the canonical order used by the model feature index."""
    return ["".join(p) for p in product(ORIGINAL_NUCLEOTIDES, repeat=kmer_size)]


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
