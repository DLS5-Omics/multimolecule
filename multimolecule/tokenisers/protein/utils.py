# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import List, Sequence

import torch

from ..alphabet import Alphabet
from ..utils import SPECIAL_TOKENS_MAP
from ..utils import convert_word_embeddings as convert_word_embeddings_

torch.manual_seed(1013)


def get_vocab_list(tokens: List[str] | None = None):
    if tokens is None:
        tokens = VOCAB_LIST
    return Alphabet(tokens).vocablulary


def get_vocab_mapping():
    return VOCAB_MAPPING


def get_special_tokens_map():
    return SPECIAL_TOKENS_MAP


def get_tokenizer_config(add_special_tokens: bool = False):
    config = TOKENIZER_CONFIG
    if add_special_tokens:
        config.setdefault("added_tokens_decoder", {})
        for i, v in enumerate(SPECIAL_TOKENS_MAP.values()):
            config["added_tokens_decoder"][str(i)] = v  # type: ignore[index]
    return config


def convert_word_embeddings(
    *old_embeddings: torch.Tensor,
    old_vocab: List[str],
    new_vocab: List[str],
    mean: float = 0.0,
    std: float = 0.02,
    vocab_mapping: dict[str, str] | None = None,
    seed: int | None = 1013,
) -> Sequence[torch.Tensor]:
    if vocab_mapping is None:
        vocab_mapping = get_vocab_mapping()
    return convert_word_embeddings_(
        *old_embeddings,
        old_vocab=old_vocab,
        new_vocab=new_vocab,
        mean=mean,
        std=std,
        vocab_mapping=vocab_mapping,
        seed=seed,
    )


VOCAB_LIST = [
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
    "V",
    "W",
    "Y",
    "X",
    "B",
    "Z",
    "J",
    "U",
    "O",
    ".",
    "*",
    "-",
]

VOCAB_MAPPING = {
    "X": "ACDEFGHIKLMNPQRSTVWY",
    "B": "DN",
    "Z": "EQ",
    "J": "IL",
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "RnaTokenizer",
    "clean_up_tokenization_spaces": True,
}
