import pytest
import torch

from multimolecule import CaLmConfig, HyenaDnaConfig


@pytest.fixture
def calm_config():
    return CaLmConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
    )


@pytest.fixture
def calm_attention_config():
    return CaLmConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
    )


@pytest.fixture
def hyenadna_config():
    return HyenaDnaConfig(
        vocab_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=32,
        filter_order=16,
        short_filter_order=3,
        num_inner_mlps=1,
        pad_vocab_size_multiple=1,
    )


@pytest.fixture
def input_ids():
    return torch.tensor([[1, 6, 7, 2]])


@pytest.fixture
def padded_input_ids():
    return torch.tensor([[1, 6, 7, 2, 0, 0]])


@pytest.fixture
def padded_attention_mask():
    return torch.tensor([[1, 1, 1, 1, 0, 0]])
