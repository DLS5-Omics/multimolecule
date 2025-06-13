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

from typing import Tuple

import torch
from torch import Tensor, nn

from .registry import PositionEmbeddingRegistry, PositionEmbeddingRegistryHF


@PositionEmbeddingRegistry.register("rotary")
@PositionEmbeddingRegistryHF.register("rotary")
class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer).

    Query and keys are transformed by rotation
    matrices which depend on their relative positions.

    Tip: **Cache**
        The inverse frequency buffer is cached and updated only when the sequence length changes or the device changes.

    Success: **Sequence Length**
        Rotary Embedding is irrespective of the sequence length and can be used for any sequence length.
    """

    def __init__(self, embedding_dim: int, base: float = 10000.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, embedding_dim, 2, dtype=dtype) / embedding_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        self._update_cos_sin_tables(k, seq_len_dim=-2)

        return (self.apply_rotary_pos_emb(q), self.apply_rotary_pos_emb(k))

    def _update_cos_sin_tables(self, x, seq_len_dim=2):
        seq_length = x.shape[seq_len_dim]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_length != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_length
            t = torch.arange(x.shape[seq_len_dim], device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, x):
        cos = self._cos_cached[:, :, : x.shape[-2], :]
        sin = self._sin_cached[:, :, : x.shape[-2], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
