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

from typing import List

import torch

from ..alphabet import Alphabet
from ..utils import SPECIAL_TOKENS_MAP

torch.manual_seed(1016)


def get_alphabet(alphabet: List[str] | str | None = None, nmers: int = 1) -> Alphabet:
    if alphabet is None:
        alphabet = STANDARD_ALPHABET if nmers <= 1 else STREAMLINE_ALPHABET
    elif isinstance(alphabet, str):
        alphabet = ALPHABETS[alphabet]
    return Alphabet(alphabet, nmers=nmers)


def get_special_tokens_map():
    return SPECIAL_TOKENS_MAP


def get_tokenizer_config(add_special_tokens: bool = False):
    config = TOKENIZER_CONFIG
    if add_special_tokens:
        config.setdefault("added_tokens_decoder", {})
        for i, v in enumerate(SPECIAL_TOKENS_MAP.values()):
            config["added_tokens_decoder"][str(i)] = v  # type: ignore[index]
    return config


STANDARD_ALPHABET = list(".()+,[]{}|<>-_:~$@^%*")

EXTENDED_ALPHABET = list(".()+,[]{}|<>")

STREAMLINE_ALPHABET = list(".()+")


ALPHABETS = {
    "standard": STANDARD_ALPHABET,
    "extended": EXTENDED_ALPHABET,
    "streamline": STREAMLINE_ALPHABET,
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "SecondaryStructureTokenizer",
    "clean_up_tokenization_spaces": True,
}
