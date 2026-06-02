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

import math

import torch
from torch import Tensor, nn

from .registry import POSITION_EMBEDDINGS, POSITION_EMBEDDINGS_HF


@POSITION_EMBEDDINGS.register("alibi")
@POSITION_EMBEDDINGS_HF.register("alibi")
class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    ALiBi adds a per-head linear penalty on the query-key distance directly to the attention
    scores: ``score[h, i, j] -= slope[h] * |i - j|``. Unlike :class:`RotaryEmbedding` (a per-token
    transform on the query/key) ALiBi is a bias on the *score matrix*, so the NestedTensor-native
    form is a [FlexAttention](https://pytorch.org/blog/flexattention/) ``score_mod`` rather than a
    materialized ``[seq, seq]`` tensor.

    The module exposes the per-head ``slopes``; consumers apply the bias from those slopes. The
    NestedTensor-native flash path passes them as ``alibi_slopes`` (the fused kernel applies
    ``-slope[h] * |i - j|`` directly); :meth:`build_alibi_tensor` materializes the equivalent dense
    ``[1, num_heads, seq, seq]`` additive bias for paths that take an explicit ``attn_mask``.

    Example:
        >>> alibi = ALiBi(num_heads=8)
        >>> bias = alibi.build_alibi_tensor(16)
        >>> bias.shape
        torch.Size([1, 8, 16, 16])
    """

    _is_hf_initialized = True

    def __init__(self, num_heads: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(self.get_slopes(num_heads), device=device, dtype=dtype)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def get_slopes(num_heads: int) -> list[float]:
        """Geometric sequence of per-head slopes, matching the original ALiBi paper."""

        def slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return slopes_power_of_2(num_heads)
        closest = 2 ** math.floor(math.log2(num_heads))
        extra = ALiBi.get_slopes(2 * closest)[0::2][: num_heads - closest]
        return slopes_power_of_2(closest) + extra

    def build_alibi_tensor(
        self, size: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Tensor:
        """Dense ``[1, num_heads, size, size]`` additive ALiBi bias for SDPA / eager fallbacks."""
        slopes = self.slopes.to(device=device, dtype=dtype)
        position = torch.arange(size, device=slopes.device)
        relative = (position[None, :] - position[:, None]).abs().to(slopes.dtype)  # (size, size)
        return (-slopes[:, None, None] * relative).unsqueeze(0)  # (1, num_heads, size, size)
