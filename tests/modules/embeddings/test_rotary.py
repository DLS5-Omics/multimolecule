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

import pytest
import torch

from multimolecule.modules.embeddings.rotary import RotaryEmbedding


class TestRotaryEmbedding:

    def test_forward(self):
        embedding_dim = 64
        base = 10000.0
        rotary_emb = RotaryEmbedding(embedding_dim, base)

        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        q_rot, k_rot = rotary_emb(q, k)

        assert k_rot.shape == k.shape

        assert rotary_emb._seq_len_cached == seq_len
        assert rotary_emb._cos_cached is not None
        assert rotary_emb._sin_cached is not None
        assert rotary_emb._cos_cached.shape == (1, 1, seq_len, head_dim)
        assert rotary_emb._sin_cached.shape == (1, 1, seq_len, head_dim)

    def test_cache_reuse(self):
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        rotary_emb(q, k)

        cos_cached_before = rotary_emb._cos_cached.clone()
        sin_cached_before = rotary_emb._sin_cached.clone()

        q2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        rotary_emb(q2, k2)

        assert torch.equal(rotary_emb._cos_cached, cos_cached_before)
        assert torch.equal(rotary_emb._sin_cached, sin_cached_before)

    def test_cache_update(self):
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, head_dim = 2, 8, 64

        seq_len1 = 10
        q1 = torch.randn(batch_size, num_heads, seq_len1, head_dim)
        k1 = torch.randn(batch_size, num_heads, seq_len1, head_dim)
        rotary_emb(q1, k1)

        seq_len2 = 15
        q2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
        k2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
        rotary_emb(q2, k2)

        assert rotary_emb._seq_len_cached == seq_len2
        assert rotary_emb._cos_cached.shape == (1, 1, seq_len2, head_dim)
        assert rotary_emb._sin_cached.shape == (1, 1, seq_len2, head_dim)

    def test_rotate_half_function(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x_rotated = RotaryEmbedding.rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(x_rotated, expected)

    def test_position_relationships(self):
        seq_len = 8
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, head_dim = 1, 1, embedding_dim
        q_test = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k_test = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_rot_test, k_rot_test = rotary_emb(q_test, k_test)

        scores_original = torch.matmul(q_test, k_test.transpose(-2, -1))
        scores_rotated = torch.matmul(q_rot_test, k_rot_test.transpose(-2, -1))

        assert not torch.allclose(scores_original, scores_rotated, atol=1e-6)

    @pytest.mark.parametrize("base", [1000.0, 10000.0, 100000.0])
    def test_base(self, base: float):
        embedding_dim = 64

        rotary_emb = RotaryEmbedding(embedding_dim, base=base)

        expected_inv_freq = 1.0 / (base ** (torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim))
        assert torch.allclose(rotary_emb.inv_freq, expected_inv_freq)

        q = torch.randn(1, 1, 5, embedding_dim)
        k = torch.randn(1, 1, 5, embedding_dim)

        q_rot, k_rot = rotary_emb(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_float16(self):
        embedding_dim = 8
        rotary_emb_fp32 = RotaryEmbedding(embedding_dim, dtype=torch.float32)
        rotary_emb_fp16 = RotaryEmbedding(embedding_dim, dtype=torch.float16)

        assert rotary_emb_fp16.inv_freq.dtype == torch.float16

        q = torch.randn(1, 1, 5, embedding_dim)
        k = torch.randn(1, 1, 5, embedding_dim)

        q_rot_fp16, k_rot_fp16 = rotary_emb_fp16(q.half(), k.half())
        assert q_rot_fp16.dtype == torch.float16
        assert k_rot_fp16.dtype == torch.float16

        q_rot_fp32, k_rot_fp32 = rotary_emb_fp32(q, k)

        assert torch.allclose(q_rot_fp16.float(), q_rot_fp32, atol=1e-4, rtol=1e-2)
        assert torch.allclose(k_rot_fp16.float(), k_rot_fp32, atol=1e-4, rtol=1e-2)
