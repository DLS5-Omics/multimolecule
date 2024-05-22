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

from itertools import product
from typing import List


class Alphabet:
    prepend_tokens: List[str] = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"]
    append_tokens: List[str] = []
    tokens: List[str]
    nmers: int

    def __init__(
        self,
        tokens: List[str],
        prepend_tokens: List[str] | None = None,
        append_tokens: List[str] | None = None,
        nmers: int = 1,
    ):
        self.tokens = tokens
        if prepend_tokens:
            self.prepend_tokens = prepend_tokens
        if append_tokens:
            self.append_tokens = append_tokens
        self.nmers = nmers

    @property
    def vocablulary(self) -> List[str]:
        return self.prepend_tokens + generate_kmer_vocabulary(self.tokens, self.nmers) + self.append_tokens


def generate_kmer_vocabulary(vocabulary: List[str], nmers: int = 1) -> List[str]:
    """
    Generates a kmer vocabulary given an original vocabulary and the size of kmers.

    Args:
        vocabulary (List[str]): The original vocabulary.
        nmers (int, defaults to 1): The size of the kmers to generate.

    Returns:
        vocabulary (List[str]): The kmer vocabulary.
    """

    if nmers <= 1:
        return vocabulary

    special_tokens, tokens = [], []
    for token in vocabulary:
        if token.startswith("<") or token.startswith("["):
            special_tokens.append(token)
        else:
            tokens.append(token)

    tokens = ["".join(kmer) for kmer in product(tokens, repeat=nmers)]

    return special_tokens + tokens
