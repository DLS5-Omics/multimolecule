# MultiMolecule
# Copyright (C) 2024-Present MultiMolecule

import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda:0", id="cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ]
)
def device(request):
    return torch.device(request.param)
