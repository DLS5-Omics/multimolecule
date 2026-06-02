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

import danling as dl
import pytest
import torch

from multimolecule.modules.embeddings.rotary import RotaryEmbedding


class TestRotaryEmbedding:

    def test_forward(self):
        embedding_dim = 64
        base = 10000.0
        rotary_emb = RotaryEmbedding(embedding_dim, base)

        batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 64
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)
        q_rot, k_rot = rotary_emb(q, k)

        assert k_rot.shape == k.shape

        assert rotary_emb._seq_len_cached == seq_length
        assert rotary_emb._cos_cached is not None
        assert rotary_emb._sin_cached is not None
        assert rotary_emb._cos_cached.shape == (1, 1, seq_length, head_dim)
        assert rotary_emb._sin_cached.shape == (1, 1, seq_length, head_dim)

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_nested_matches_dense(self, interleaved):
        embedding_dim, num_heads = 64, 4
        lengths = [7, 11, 5]
        rotary_emb = RotaryEmbedding(embedding_dim, interleaved=interleaved)
        q = dl.NestedTensor([torch.randn(num_heads, length, embedding_dim) for length in lengths])
        k = dl.NestedTensor([torch.randn(num_heads, length, embedding_dim) for length in lengths])

        q_rot, k_rot = rotary_emb(q, k)

        for index in range(len(lengths)):
            dense = RotaryEmbedding(embedding_dim, interleaved=interleaved)
            q_ref, k_ref = dense(q[index].unsqueeze(0), k[index].unsqueeze(0))
            torch.testing.assert_close(q_rot[index], q_ref[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(k_rot[index], k_ref[0], atol=1e-5, rtol=1e-5)

    def test_cache_reuse(self):
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, seq_length, head_dim = 2, 8, 10, 64
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)
        rotary_emb(q, k)

        cos_cached_before = rotary_emb._cos_cached.clone()
        sin_cached_before = rotary_emb._sin_cached.clone()

        q2 = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k2 = torch.randn(batch_size, num_heads, seq_length, head_dim)
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

    def test_interleaved_rotate_half_function(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x_rotated = RotaryEmbedding.rotate_half(x, interleaved=True)
        expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
        assert torch.allclose(x_rotated, expected)

    def test_interleaved_matches_complex_pair_rotation(self):
        embedding_dim = 8
        seq_length = 5
        rotary_emb = RotaryEmbedding(embedding_dim, interleaved=True)

        q = torch.randn(2, 3, seq_length, embedding_dim)
        k = torch.randn(2, 3, seq_length, embedding_dim)
        q_rot, k_rot = rotary_emb(q, k)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        freqs = torch.outer(torch.arange(seq_length).float(), inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        while freqs_cis.ndim < q.ndim:
            freqs_cis = freqs_cis.unsqueeze(0)

        q_expected = torch.view_as_real(torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2)) * freqs_cis)
        k_expected = torch.view_as_real(torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2)) * freqs_cis)

        assert torch.allclose(q_rot, q_expected.flatten(-2), atol=1e-6)
        assert torch.allclose(k_rot, k_expected.flatten(-2), atol=1e-6)

    def test_position_relationships(self):
        seq_length = 8
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, head_dim = 1, 1, embedding_dim
        q_test = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k_test = torch.randn(batch_size, num_heads, seq_length, head_dim)

        q_rot_test, k_rot_test = rotary_emb(q_test, k_test)

        scores_original = torch.matmul(q_test, k_test.transpose(-2, -1))
        scores_rotated = torch.matmul(q_rot_test, k_rot_test.transpose(-2, -1))

        assert not torch.allclose(scores_original, scores_rotated, atol=1e-6)

    @pytest.mark.parametrize("base", [1000.0, 10000.0, 100000.0])
    def test_base(self, base: float):
        embedding_dim = 64

        rotary_emb = RotaryEmbedding(embedding_dim, base=base)

        expected_inv_freq = 1.0 / (base ** (torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim))
        inv_freq: torch.Tensor = rotary_emb.inv_freq  # type: ignore[assignment]
        assert torch.allclose(inv_freq, expected_inv_freq)

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

        assert torch.allclose(q_rot_fp16.float(), q_rot_fp32, atol=3e-3, rtol=1e-2)
        assert torch.allclose(k_rot_fp16.float(), k_rot_fp32, atol=3e-3, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_preserves_input_dtype(self, dtype: torch.dtype):
        """A default (fp32-table) module fed low-precision inputs must return that input dtype.

        This is the production path: every consumer builds ``RotaryEmbedding(head_dim)``
        with the default fp32 ``dtype`` for table stability, then runs the model in
        bf16/fp16. Rotary must not leak fp32 into q/k, otherwise q/k mismatch v at the
        attention kernel.
        """
        embedding_dim = 8
        rotary_emb = RotaryEmbedding(embedding_dim, interleaved=True)
        assert rotary_emb.inv_freq.dtype == torch.float32

        q = torch.randn(1, 1, 5, embedding_dim, dtype=dtype)
        k = torch.randn(1, 1, 5, embedding_dim, dtype=dtype)
        q_rot, k_rot = rotary_emb(q, k)

        assert q_rot.dtype == dtype
        assert k_rot.dtype == dtype

    @pytest.mark.parametrize("scale", [1.0, 2.0, 4.0, 0.5])
    def test_scale(self, scale: float):
        """Test that scale parameter affects the rotary embeddings."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim, scale=scale)

        batch_size, num_heads, seq_length, head_dim = 1, 1, 10, embedding_dim
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)

        q_rot, k_rot = rotary_emb(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Different scales should produce different results
        if scale != 1.0:
            rotary_emb_default = RotaryEmbedding(embedding_dim, scale=1.0)
            q_rot_default, k_rot_default = rotary_emb_default(q, k)
            assert not torch.allclose(q_rot, q_rot_default, atol=1e-6)
            assert not torch.allclose(k_rot, k_rot_default, atol=1e-6)

    def test_scale_default(self):
        """Test that default scale is 1.0."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)
        assert rotary_emb.scale == 1.0

    def test_offset_zero(self):
        """Test that offset=0 (default) works correctly."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, seq_length, head_dim = 1, 1, 10, embedding_dim
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)

        # offset=0 should be the same as no offset
        q_rot1, k_rot1 = rotary_emb(q, k)
        q_rot2, k_rot2 = rotary_emb(q, k, offset=0)

        assert torch.allclose(q_rot1, q_rot2, atol=1e-6)
        assert torch.allclose(k_rot1, k_rot2, atol=1e-6)

    def test_offset_with_seq_len(self):
        """Test that offset works correctly with seq_length parameter."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, seq_length, head_dim = 1, 1, 5, embedding_dim
        offset = 10
        full_seq_len = offset + seq_length

        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)

        # Apply with offset
        q_rot, k_rot = rotary_emb(q, k, offset=offset, seq_length=full_seq_len)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert rotary_emb._seq_len_cached == full_seq_len

    def test_offset_without_seq_len_error(self):
        """Test that offset > 0 without seq_length raises ValueError."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, seq_length, head_dim = 1, 1, 5, embedding_dim
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)

        with pytest.raises(ValueError, match="seq_length must be provided when offset > 0"):
            rotary_emb(q, k, offset=10)

    def test_offset_consistency(self):
        """Test that offset produces consistent results with manual slicing."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, full_seq_len, head_dim = 1, 1, 20, embedding_dim
        offset = 5
        seq_length = 10

        # Create full sequence
        q_full = torch.randn(batch_size, num_heads, full_seq_len, head_dim)
        k_full = torch.randn(batch_size, num_heads, full_seq_len, head_dim)

        # Apply to full sequence
        q_rot_full, k_rot_full = rotary_emb(q_full, k_full)

        # Apply with offset to a slice
        q_slice = q_full[:, :, offset : offset + seq_length, :]
        k_slice = k_full[:, :, offset : offset + seq_length, :]
        q_rot_slice, k_rot_slice = rotary_emb(q_slice, k_slice, offset=offset, seq_length=full_seq_len)

        # Results should match
        assert torch.allclose(q_rot_full[:, :, offset : offset + seq_length, :], q_rot_slice, atol=1e-5)
        assert torch.allclose(k_rot_full[:, :, offset : offset + seq_length, :], k_rot_slice, atol=1e-5)

    def test_scale_with_offset(self):
        """Test that scale and offset work together."""
        embedding_dim = 64
        scale = 2.0
        rotary_emb = RotaryEmbedding(embedding_dim, scale=scale)

        batch_size, num_heads, seq_length, head_dim = 1, 1, 5, embedding_dim
        offset = 10
        full_seq_len = offset + seq_length

        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)

        q_rot, k_rot = rotary_emb(q, k, offset=offset, seq_length=full_seq_len)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert rotary_emb.scale == scale
        assert rotary_emb._seq_len_cached == full_seq_len

    def test_seq_len_auto_detection(self):
        """Test that seq_length is automatically detected when not provided."""
        embedding_dim = 64
        rotary_emb = RotaryEmbedding(embedding_dim)

        batch_size, num_heads, seq_length, head_dim = 1, 1, 15, embedding_dim
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)

        q_rot, k_rot = rotary_emb(q, k)

        assert rotary_emb._seq_len_cached == seq_length
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
