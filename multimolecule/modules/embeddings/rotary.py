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
from danling import NestedTensor
from torch import Tensor, nn

from .registry import POSITION_EMBEDDINGS, POSITION_EMBEDDINGS_HF


@POSITION_EMBEDDINGS.register("rotary")
@POSITION_EMBEDDINGS_HF.register("rotary")
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
        Use the `scale` parameter to extend context length beyond training (e.g., scale=2.0 doubles effective context).

    Example:
        >>> embedding = RotaryEmbedding(embedding_dim=64)
        >>> query, key = torch.randn(2, 4, 28, 64), torch.randn(2, 4, 28, 64)
        >>> query, key = embedding(query, key)
        >>> query.shape
        torch.Size([2, 4, 28, 64])
        >>> # For extended context length
        >>> embedding_extended = RotaryEmbedding(embedding_dim=64, scale=2.0)
        >>> embedding.state_dict()  # no weight in state_dict
        OrderedDict()
    """

    _is_hf_initialized = True
    _seq_len_cached: int | None = None
    _cos_cached: Tensor | None = None
    _sin_cached: Tensor | None = None

    def __init__(
        self,
        embedding_dim: int,
        base: float = 10000.0,
        scale: float = 1.0,
        interleaved: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize rotary position embeddings.

        Args:
            embedding_dim: Dimension of the embeddings (must be even)
            base: Base for computing inverse frequencies. Defaults to 10000.0.
            scale: Scaling factor for frequencies. Values > 1.0 extend context length
                   (e.g., scale=2.0 doubles the effective context). Defaults to 1.0.
            interleaved: Whether to rotate adjacent feature pairs. Defaults to False, which preserves the
                   existing split-half layout.
            dtype: Data type for computations. Defaults to torch.float32.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base = base
        self.scale = scale
        self.interleaved = interleaved
        inv_freq_exponent = torch.arange(0, self.embedding_dim, 2, device=device, dtype=dtype) / self.embedding_dim
        self.register_buffer("inv_freq", 1.0 / (self.base**inv_freq_exponent), persistent=False)
        self._initialized = False

    def forward(self, q: Tensor, k: Tensor, offset: int = 0, seq_length: int | None = None) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape `(batch_size, num_heads, seq_length, embedding_dim)`
            k: Key tensor of shape `(batch_size, num_heads, seq_length, embedding_dim)`
            offset: Position offset for the start of the sequence (used with past_key_values).
                    Defaults to 0.
            seq_length: Full sequence length including offset. If None, uses the sequence length
                    from the input tensors. Required when offset > 0.

        Returns:
            Tuple of (rotated_query, rotated_key) tensors with the same shapes as inputs.
        """
        if not self._initialized:
            inv_freq_exponent = (
                torch.arange(0, self.embedding_dim, 2, device=q.device, dtype=self.inv_freq.dtype) / self.embedding_dim
            )
            self.register_buffer("inv_freq", 1.0 / (self.base**inv_freq_exponent), persistent=False)
            self._initialized = True
        if offset > 0 and seq_length is None:
            raise ValueError("seq_length must be provided when offset > 0")

        if isinstance(q, NestedTensor):
            reference = q[..., 0, :, 0]
            positions = torch.cumsum(torch.ones_like(reference, dtype=torch.long), dim=-1) - 1 + offset
            cos, sin = self.cos_sin_tables(positions, dtype=q.dtype)
            return (
                self.apply_rotary(q, cos, sin, interleaved=self.interleaved),
                self.apply_rotary(k, cos, sin, interleaved=self.interleaved),
            )

        if seq_length is None:
            seq_length = k.shape[-2]

        self._update_cos_sin_tables(k, seq_len_dim=-2, seq_length=seq_length)
        return self.apply_rotary_pos_emb(q, offset=offset), self.apply_rotary_pos_emb(k, offset=offset)

    def _update_cos_sin_tables(
        self, x: Tensor, seq_len_dim: int = 2, seq_length: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Update cached cos/sin tables for rotary embeddings.

        Args:
            x: Input tensor to determine device and dtype
            seq_len_dim: Dimension containing sequence length (default: -2)
            seq_length: Full sequence length to cache. If None, uses x.shape[seq_len_dim]
        """
        if seq_length is None:
            seq_length = x.shape[seq_len_dim]

        needs_update = (
            seq_length != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device
        )
        if needs_update:
            self._seq_len_cached = seq_length
            t = torch.arange(seq_length, device=x.device, dtype=self.inv_freq.dtype)
            # Apply scaling: divide frequencies by scale to extend context length
            freqs = torch.outer(t, self.inv_freq) / self.scale
            if self.interleaved:
                emb = freqs.repeat_interleave(2, dim=-1).to(x.device)
            else:
                emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        # At this point, _cos_cached and _sin_cached are guaranteed to be Tensor
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Apply rotary position embeddings to a tensor.

        Args:
            x: Input tensor of shape `(batch_size, num_heads, seq_length, embedding_dim)`
            offset: Position offset for the start of the sequence (used with past_key_values).
                    Defaults to 0.

        Returns:
            Rotated tensor with the same shape as input.
        """
        if self._cos_cached is None or self._sin_cached is None:
            raise RuntimeError("Cos/sin tables not initialized. Call forward() or _update_cos_sin_tables() first.")

        cos = self._cos_cached[:, :, offset : offset + x.shape[-2], :]
        sin = self._sin_cached[:, :, offset : offset + x.shape[-2], :]
        rotated = (x * cos) + (self.rotate_half(x, interleaved=self.interleaved) * sin)
        return rotated.type_as(x)

    @staticmethod
    def rotate_half(x: Tensor, interleaved: bool = False) -> Tensor:
        if interleaved:
            return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        half = x.shape[-1] // 2
        return torch.stack((-x[..., half:], x[..., :half]), dim=-2).flatten(-2)

    def cos_sin_tables(self, position_ids: Tensor, dtype: torch.dtype | None = None) -> Tuple[Tensor, Tensor]:
        """Compute ``(cos, sin)`` rotary tables for explicit ``position_ids``.

        Unlike :meth:`forward`, positions come from ``position_ids`` rather than an internal
        ``arange`` over the sequence. This is what autoregressive models (KV-cache offsets,
        generation) and packed ``NestedTensor`` inputs (per-sequence positions) require. Works for
        dense ``position_ids`` of shape ``(batch, seq)`` and ragged ``NestedTensor`` positions alike;
        the result broadcasts over the head dimension and is *not* cached (positions vary per call).

        Args:
            position_ids: Integer positions, shape ``(batch, seq)`` (dense or ``NestedTensor``).
            dtype: Optional output dtype for ``cos``/``sin``. Frequencies are always accumulated in the
                   ``inv_freq`` dtype (float32 by default) for numerical stability, then cast.

        Returns:
            Tuple of ``(cos, sin)`` each shaped ``(batch, 1, seq, embedding_dim)``.
        """
        if not self._initialized:
            inv_freq_exponent = (
                torch.arange(0, self.embedding_dim, 2, device=position_ids.device, dtype=self.inv_freq.dtype)
                / self.embedding_dim
            )
            self.register_buffer("inv_freq", 1.0 / (self.base**inv_freq_exponent), persistent=False)
            self._initialized = True
        freqs = position_ids.unsqueeze(-1).to(self.inv_freq.dtype) * self.inv_freq / self.scale
        emb = torch.repeat_interleave(freqs, 2, dim=-1) if self.interleaved else torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin

    @staticmethod
    def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool = False) -> Tensor:
        """Apply precomputed ``cos``/``sin`` tables to ``x`` (no internal cache).

        Companion to :meth:`cos_sin_tables` for models that precompute the tables once at the model level and
        apply them inside attention (e.g. autoregressive models passing ``position_embeddings`` down).
        """
        return (x * cos) + (RotaryEmbedding.rotate_half(x, interleaved=interleaved) * sin)
