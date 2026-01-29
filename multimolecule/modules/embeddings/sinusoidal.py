# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
# Copyright (C) 2020 The Facebook AI Research Team Authors
# Copyright (C) 2020 The HuggingFace Inc. team.

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

import math

import torch
from torch import Tensor, nn

from .registry import POSITION_EMBEDDINGS, POSITION_EMBEDDINGS_HF


@POSITION_EMBEDDINGS.register("sinusoidal")
@POSITION_EMBEDDINGS_HF.register("sinusoidal")
class SinusoidalEmbedding(nn.Embedding):
    r"""
    Sinusoidal positional embeddings for inputs with any length.

    Note: **Freezing**
        The embeddings are frozen and cannot be trained.
        They will not be saved in the model's state_dict.

    Tip: **Padding Idx**
        Padding symbols are ignored if the padding_idx is specified.

    Success: **Sequence Length**
        These embeddings get automatically extended in forward if more positions is needed.

    Args:
        num_embeddings: The number of embeddings to use.
        embedding_dim: The dimension of the embeddings.
        padding_idx: The index of the padding symbol.
        bias: The bias of the embeddings.

    Example:
        >>> embedding = SinusoidalEmbedding(num_embeddings=128, embedding_dim=64)
        >>> input_ids = torch.arange(28).repeat(4).view(4, -1)
        >>> input_embeds = torch.randn(4, 28, 64)
        >>> embeddings = embedding(input_ids)
        >>> embeddings.shape  # no batch dimension if padding_idx is None
        torch.Size([28, 64])
        >>> input_embeds = input_embeds + embeddings
        >>> input_embeds.shape
        torch.Size([4, 28, 64])
        >>> embedding = SinusoidalEmbedding(num_embeddings=128, embedding_dim=64, padding_idx=0)
        >>> embeddings = embedding(input_ids)
        >>> embeddings.shape  # batch dimension if padding_idx is not None
        torch.Size([4, 28, 64])
        >>> embedding.state_dict()  # no weight in state_dict
        OrderedDict()
    """

    _is_hf_initialized = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        bias: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        weight = self.get_embedding(num_embeddings, embedding_dim, padding_idx, device=device, dtype=dtype)
        super().__init__(num_embeddings, embedding_dim, padding_idx, _weight=weight.detach(), _freeze=True, **kwargs)
        del self.weight
        self.register_buffer("weight", weight, persistent=False)
        self.bias = bias
        self._initialized = False

    def update_weight(self, num_embeddings: int):
        weight = self.get_embedding(
            num_embeddings, self.embedding_dim, self.padding_idx, dtype=self.weight.dtype, device=self.weight.device
        )
        self.register_buffer("weight", weight, persistent=False)

    @staticmethod
    def get_embedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        if device is None:
            device = get_default_device()
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -(math.log(10000) / (half_dim - 1)))
        emb = torch.arange(num_embeddings, device=device, dtype=dtype).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1, dtype=dtype, device=device)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.detach()

    @staticmethod
    def get_position_ids(tensor: Tensor, padding_idx: int | None = None):
        """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        if padding_idx is None:
            return torch.cumsum(tensor.new_ones(tensor.size(1), dtype=torch.long), dim=0) - 1
        mask = tensor.ne(padding_idx).long()
        return torch.cumsum(mask, dim=1, dtype=torch.long) * mask + padding_idx

    def forward(self, input_ids: Tensor) -> Tensor:
        if not self._initialized:
            self.update_weight(self.num_embeddings)
            self._initialized = True
        _, seq_length = input_ids.shape[:2]
        # expand embeddings if needed
        max_position = seq_length + self.bias + 1
        if self.padding_idx is not None:
            max_position += self.padding_idx
        if max_position > self.weight.size(0):
            self.update_weight(max_position)
        # Need to shift the position ids by the padding index
        position_ids = self.get_position_ids(input_ids, self.padding_idx) + self.bias
        return super().forward(position_ids)


def get_default_device() -> torch.device:
    try:
        return torch.get_default_device()
    except ArithmeticError:
        return torch.device("cpu")
