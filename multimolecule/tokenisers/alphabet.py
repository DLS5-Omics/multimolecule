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

from functools import lru_cache
from itertools import product
from typing import Sequence, Tuple


class Alphabet:
    prepend_tokens: Tuple[str, ...] = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")
    append_tokens: Tuple[str, ...] = ()
    tokens: Tuple[str, ...]
    nmers: int

    def __init__(
        self,
        tokens: Sequence[str],
        prepend_tokens: Tuple[str, ...] | None = None,
        append_tokens: Tuple[str, ...] | None = None,
        nmers: int = 1,
    ):
        if isinstance(tokens, Alphabet):
            tokens = tokens.tokens
        self.tokens = tuple(tokens)
        if prepend_tokens:
            self.prepend_tokens = tuple(prepend_tokens)
        if append_tokens:
            self.append_tokens = tuple(append_tokens)
        self.nmers = nmers

    @property
    def vocabulary(self) -> Tuple[str, ...]:
        return self._vocabulary(self.prepend_tokens, self.tokens, self.nmers, self.append_tokens)

    @staticmethod
    @lru_cache(maxsize=None)
    def _vocabulary(
        prepend_tokens: Tuple[str, ...], tokens: Tuple[str, ...], nmers: int, append_tokens: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        return prepend_tokens + generate_kmer_vocabulary(tokens, nmers) + append_tokens

    def __iter__(self):
        return iter(self.vocabulary)

    def __len__(self):
        return len(self.vocabulary)

    def __contains__(self, item: str):
        return item in self.vocabulary

    def __repr__(self) -> str:
        repr_parts = [f"Alphabet(tokens={self.tokens}"]
        if self.nmers > 1:
            repr_parts.append(f"nmers={self.nmers}")
        repr_parts.append(f"prepend_tokens={self.prepend_tokens}")
        repr_parts.append(f"append_tokens={self.append_tokens})")
        return ", ".join(repr_parts)


def generate_kmer_vocabulary(vocabulary: Tuple[str, ...], nmers: int = 1) -> Tuple[str, ...]:
    """
    Generates a kmer vocabulary given an original vocabulary and the size of kmer.

    Args:
        vocabulary (List[str]): The original vocabulary.
        nmers (int, defaults to 1): The size of kmer to generate.

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

    return tuple(special_tokens) + tuple("".join(kmer) for kmer in product(tokens, repeat=nmers))
