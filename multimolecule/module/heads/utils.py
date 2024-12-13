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

import torch
from torch import Tensor


def unfold_kmer_embeddings(
    embeddings: Tensor,
    attention_mask: Tensor,
    nmers: int,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> Tensor:
    r"""
    Unfold k-mer embeddings to token embeddings.

    For k-mer input, each embedding column represents k tokens.
    This should be fine for sequence level tasks, but sacrifices the resolution for token level tasks.
    This function unfolds the k-mer embeddings to token embeddings by sliding averaging the k-mer embeddings.

    For example:

    input tokens = `ACGU`

    2-mer embeddings = `[<CLS>, AC, CG, GU, <SEP>]`.

    token embeddings = `[<CLS>, AC, (AC + CG) / 2, (CG + GU) / 2, GU, <SEP>]`.

    Args:
        embeddings: The k-mer embeddings.
        attention_mask: The attention mask.
        nmers: The number of tokens in each k-mer.
        bos_token_id: The id of the beginning of sequence token.
            If not None, the first valid token will not be included in sliding averaging.
        eos_token_id: The id of the end of sequence token.
            If not None, the last valid token will not be included in sliding averaging.

    Returns:
        The token embeddings.

    Examples:
        >>> from danling import NestedTensor
        >>> embeddings = NestedTensor(torch.arange(3).repeat(2, 1).T, torch.arange(5).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 3, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.0, 2.0, 3.0, 0.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        >>> embeddings = NestedTensor(torch.arange(5).repeat(2, 1).T, torch.arange(7).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 4, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 5.0, 0.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5, 6.0, 7.0]
        >>> embeddings = NestedTensor(torch.arange(7).repeat(2, 1).T, torch.arange(11).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 5, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0]
        >>> embeddings = NestedTensor(torch.arange(3).repeat(2, 1).T, torch.arange(4).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 6, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 4.0]
        >>> embeddings = NestedTensor(torch.arange(1).repeat(2, 1).T, torch.arange(2).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 6)
        >>> output[0, :, 0].tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0]
    """

    batch_size, seq_length, hidden_size = embeddings.size()
    last_valid_indices = attention_mask.sum(dim=-1)
    output = torch.zeros(batch_size, seq_length + nmers - 1, hidden_size, device=embeddings.device)
    for index, (tensor, seq_length) in enumerate(zip(embeddings, last_valid_indices)):
        embedding = tensor[:seq_length]
        if bos_token_id is not None:
            embedding = embedding[1:]
        if eos_token_id is not None:
            embedding = embedding[:-1]
        if len(embedding) > nmers:
            begin = torch.stack([embedding[:i].mean(0) for i in range(1, nmers)])
            medium = embedding.unfold(0, nmers, 1).mean(-1)
            end = torch.stack([embedding[-i:].mean(0) for i in range(nmers - 1, 0, -1)])
            embedding = torch.cat([begin, medium, end])
        elif len(embedding) > 2:
            begin = torch.stack([embedding[:i].mean(0) for i in range(1, len(embedding))])
            end = torch.stack([embedding[-i:].mean(0) for i in range(nmers, 0, -1)])
            embedding = torch.cat([begin, end])
        elif len(embedding) == 2:
            medium = embedding.mean(0).repeat(nmers - 1, 1)
            embedding = torch.cat([embedding[0][None, :], medium, embedding[1][None, :]])
        elif len(embedding) == 1:
            embedding = embedding.repeat(nmers, 1)
        else:
            raise ValueError("Sequence length is less than nmers.")
        if bos_token_id is not None:
            embedding = torch.cat([tensor[0][None, :], embedding])
        if eos_token_id is not None:
            embedding = torch.cat([embedding, tensor[seq_length - 1][None, :]])
        output[index, : seq_length + nmers - 1] = embedding
    return output
