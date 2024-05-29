# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
# Copyright (C) 2020 The Facebook AI Research Team Authors
# Copyright (C) 2020 The HuggingFace Inc. team.

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

import math

import torch
import torch.onnx.operators
from torch import Tensor, nn

from .registry import PositionEmbeddingRegistry, PositionEmbeddingRegistryHF


@PositionEmbeddingRegistry.register("sinusoidal")
@PositionEmbeddingRegistryHF.register("sinusoidal")
class SinusoidalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.

    We don't want to save the weight of this embedding since it's not trained (deterministic) and it can be huge.

    Padding symbols are ignored.

    These embeddings get automatically extended in forward if more positions is needed.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        weight = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        super().__init__(num_embeddings, embedding_dim, padding_idx, _weight=weight.detach(), _freeze=True)

    def update_weight(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        weight = self.get_embedding(num_embeddings, embedding_dim, padding_idx).to(
            dtype=self.weight.dtype, device=self.weight.device  # type: ignore[has-type]
        )
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> Tensor:
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -(math.log(10000) / (half_dim - 1)))
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(self, input: Tensor):
        _, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weight.size(0):
            # expand embeddings if needed
            self.update_weight(max_pos, self.embedding_dim, self.padding_idx)
        positions = self.make_positions(input, self.padding_idx)
        return super().forward(positions)
