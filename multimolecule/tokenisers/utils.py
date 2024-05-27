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

import random
from typing import List, Sequence

import torch
from torch import Tensor


def convert_word_embeddings(
    *old_embeddings: Tensor,
    old_vocab: List[str],
    new_vocab: List[str],
    mean: float = 0.0,
    std: float = 0.02,
    vocab_mapping: dict[str, str] | None = None,
    seed: int | None = 1013,
) -> Sequence[Tensor]:
    if old_vocab == new_vocab:
        return old_embeddings
    if vocab_mapping is None:
        raise ValueError("vocab_mapping must be provided to convert the embeddings.")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    new_embeddings = []
    # Initialize the new embeddings
    for embeddings in old_embeddings:
        shape = embeddings.shape
        if shape[0] != len(old_vocab):
            raise ValueError("The first dimension of the embeddings must match the size of the vocabulary.")
        if embeddings.ndim == 1:  # Bias
            new_embeddings.append(torch.zeros(len(new_vocab)))
        else:
            new_embeddings.append(torch.normal(size=(len(new_vocab), *shape[1:]), mean=mean, std=std))

    # First Pass, copy the embeddings for the tokens that are in both vocabularies
    for old_index, old_token in enumerate(old_vocab):
        new_index = new_vocab.index(old_token)
        for new_embed, old_embed in zip(new_embeddings, old_embeddings):
            new_embed[new_index] = old_embed[old_index]

    # Second Pass, average the embeddings for the tokens that are in the new vocabulary but not in the old
    for token, tokens in vocab_mapping.items():
        if token not in new_vocab or token in old_vocab or len(tokens) == 1:
            continue
        index = new_vocab.index(token)
        indexes = [new_vocab.index(t) for t in tokens]
        for embed in new_embeddings:
            embed[index] = embed[indexes].mean(dim=0)

    return new_embeddings


SPECIAL_TOKENS_MAP = {
    "pad_token": {
        "content": "<pad>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "cls_token": {
        "content": "<cls>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "eos_token": {
        "content": "<eos>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "unk_token": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "mask_token": {
        "content": "<mask>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "null_token": {
        "content": "<null>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
}
