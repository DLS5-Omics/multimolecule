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


# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


import math

import torch

from multimolecule.modules.embeddings.sinusoidal import SinusoidalEmbedding


class TestSinusoidalEmbedding:
    """Unit tests for SinusoidalEmbedding class."""

    def test_initialization(self):
        """Test basic SinusoidalEmbedding initialization."""
        num_embeddings = 128
        embedding_dim = 64
        padding_idx = None

        sinusoidal_emb = SinusoidalEmbedding(num_embeddings, embedding_dim, padding_idx)

        assert sinusoidal_emb.num_embeddings == num_embeddings
        assert sinusoidal_emb.embedding_dim == embedding_dim
        assert sinusoidal_emb.padding_idx == padding_idx
        assert sinusoidal_emb.bias == 1
        assert sinusoidal_emb.weight.shape == (num_embeddings, embedding_dim)
        assert sinusoidal_emb.weight.requires_grad is False
        assert "weight" in sinusoidal_emb._buffers
        assert not sinusoidal_emb.state_dict()

    def test_forward_no_padding(self):
        num_embeddings = 128
        embedding_dim = 64

        sinusoidal_emb = SinusoidalEmbedding(num_embeddings, embedding_dim)

        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        batch_size, seq_length = input_ids.shape

        embeddings = sinusoidal_emb(input_ids)
        assert embeddings.shape == (seq_length, embedding_dim)

        embeddings2 = sinusoidal_emb(input_ids)
        assert torch.equal(embeddings, embeddings2)

    def test_forward_with_padding(self):
        num_embeddings = 50
        embedding_dim = 32
        padding_idx = 0

        sinusoidal_emb = SinusoidalEmbedding(num_embeddings, embedding_dim, padding_idx)

        padding_embedding = sinusoidal_emb.weight[padding_idx]
        assert torch.allclose(padding_embedding, torch.zeros(embedding_dim))

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        embeddings = sinusoidal_emb(input_ids)

        batch_size, seq_length = input_ids.shape
        assert embeddings.shape == (batch_size, seq_length, embedding_dim)

        position_ids = sinusoidal_emb.get_position_ids(input_ids, padding_idx) + sinusoidal_emb.bias

        non_padding_positions = input_ids != padding_idx
        for i, j in torch.nonzero(non_padding_positions, as_tuple=False):
            assert position_ids[i, j] > padding_idx

    def test_auto_extend(self):
        num_embeddings = 10
        embedding_dim = 16

        sinusoidal_emb = SinusoidalEmbedding(num_embeddings, embedding_dim)

        long_input_ids = torch.arange(15).unsqueeze(0)  # sequence length 15 > 10

        embeddings = sinusoidal_emb(long_input_ids)

        assert sinusoidal_emb.weight.shape[0] >= 15 + sinusoidal_emb.bias + 1
        assert embeddings.shape == (15, embedding_dim)

    def test_different_bias_values(self):
        num_embeddings = 50
        embedding_dim = 32

        for bias in [0, 1, 2, 5]:
            sinusoidal_emb = SinusoidalEmbedding(num_embeddings, embedding_dim, bias=bias)
            assert sinusoidal_emb.bias == bias

            input_ids = torch.tensor([[1, 2, 3]])
            embeddings = sinusoidal_emb(input_ids)
            assert embeddings.shape == (3, embedding_dim)

    def test_update_weight(self):
        num_embeddings = 10
        embedding_dim = 16

        sinusoidal_emb = SinusoidalEmbedding(num_embeddings, embedding_dim)
        original_weight = sinusoidal_emb.weight.clone()

        new_num_embeddings = 20
        sinusoidal_emb.update_weight(new_num_embeddings)

        assert sinusoidal_emb.weight.shape[0] == new_num_embeddings

        assert torch.allclose(sinusoidal_emb.weight[:num_embeddings], original_weight)

    def test_get_embedding(self):
        num_embeddings = 20
        embedding_dim = 8

        embeddings = SinusoidalEmbedding.get_embedding(num_embeddings, embedding_dim)
        assert embeddings.shape == (num_embeddings, embedding_dim)
        assert embeddings.requires_grad is False

    def test_get_embedding_with_padding(self):
        num_embeddings = 20
        embedding_dim = 8
        padding_idx = 0

        embeddings_padded = SinusoidalEmbedding.get_embedding(num_embeddings, embedding_dim, padding_idx)
        assert torch.allclose(embeddings_padded[padding_idx], torch.zeros(embedding_dim))

    def test_get_embedding_odd_dimension(self):
        odd_dim = 7
        embeddings_odd = SinusoidalEmbedding.get_embedding(10, odd_dim)
        assert embeddings_odd.shape == (10, odd_dim)

    def test_get_position_ids_no_padding(self):
        tensor = torch.ones(2, 5)
        position_ids = SinusoidalEmbedding.get_position_ids(tensor)
        expected = torch.tensor([0, 1, 2, 3, 4])
        assert torch.equal(position_ids, expected)

    def test_get_position_ids_with_padding(self):
        padding_idx = 0
        tensor_with_padding = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        position_ids_padded = SinusoidalEmbedding.get_position_ids(tensor_with_padding, padding_idx)

        expected_first = torch.tensor([1, 2, 3, 0, 0])
        expected_second = torch.tensor([1, 2, 0, 0, 0])

        assert torch.equal(position_ids_padded[0], expected_first)
        assert torch.equal(position_ids_padded[1], expected_second)

    def test_sinusoidal_pattern(self):
        num_embeddings = 100
        embedding_dim = 64

        embeddings = SinusoidalEmbedding.get_embedding(num_embeddings, embedding_dim)

        pos = 10
        half_dim = embedding_dim // 2

        for dim in range(0, min(half_dim, 4)):
            emb_factor = math.exp(dim * -(math.log(10000) / (half_dim - 1)))
            angle = pos * emb_factor

            expected_sin = math.sin(angle)
            expected_cos = math.cos(angle)

            assert torch.allclose(embeddings[pos, dim], torch.tensor(expected_sin), atol=1e-6)
            assert torch.allclose(embeddings[pos, dim + half_dim], torch.tensor(expected_cos), atol=1e-6)

    def test_embedding_properties(self):
        num_embeddings = 50
        embedding_dim = 32

        embeddings = SinusoidalEmbedding.get_embedding(num_embeddings, embedding_dim)

        assert torch.all(embeddings >= -1.1)
        assert torch.all(embeddings <= 1.1)

        assert not torch.allclose(embeddings[0], embeddings[1])
        assert not torch.allclose(embeddings[5], embeddings[10])

    def test_padding_preservation(self):
        num_embeddings = 20
        embedding_dim = 16
        padding_idx = 3

        embeddings1 = SinusoidalEmbedding.get_embedding(num_embeddings, embedding_dim, padding_idx)
        embeddings2 = SinusoidalEmbedding.get_embedding(num_embeddings * 2, embedding_dim, padding_idx)

        assert torch.allclose(embeddings1[padding_idx], torch.zeros(embedding_dim))
        assert torch.allclose(embeddings2[padding_idx], torch.zeros(embedding_dim))

        for i in range(num_embeddings):
            if i != padding_idx:
                assert torch.allclose(embeddings1[i], embeddings2[i])

    def test_float16(self):
        num_embeddings = 128
        embedding_dim = 8
        sinusodial_emb_fp32 = SinusoidalEmbedding(num_embeddings, embedding_dim, dtype=torch.float32)
        sinusodial_emb_fp16 = SinusoidalEmbedding(num_embeddings, embedding_dim, dtype=torch.float16)

        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        embeddings_fp16 = sinusodial_emb_fp16(input_ids)
        embeddings_fp32 = sinusodial_emb_fp32(input_ids)

        assert embeddings_fp16.dtype == torch.float16
        assert torch.allclose(embeddings_fp16.float(), embeddings_fp32, atol=1e-4, rtol=1e-2)
