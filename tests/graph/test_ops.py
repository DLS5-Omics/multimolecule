# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import pytest
import torch

from multimolecule.graph import scatter, scatter_mean


class TestScatter:
    @pytest.mark.parametrize(
        "reduce,expected,gradient",
        [
            ("sum", [2, 0, 3, 0], [1, 1, 1]),
            ("mean", [1, 0, 3, 0], [0.5, 0.5, 1]),
            ("min", [-2, 0, 3, 0], [1, 0, 1]),
            ("max", [4, 0, 3, 0], [0, 1, 1]),
        ],
    )
    def test_reductions_and_gradients(self, device, reduce, expected, gradient):
        src = torch.tensor([-2.0, 4.0, 3.0], device=device, requires_grad=True)
        result = scatter(src, torch.tensor([0, 0, 2], device=device), dim_size=4, reduce=reduce)
        torch.testing.assert_close(result, torch.tensor(expected, dtype=src.dtype, device=device))
        torch.testing.assert_close(torch.autograd.grad(result.sum(), src)[0], src.new_tensor(gradient))

    def test_integer_reduction(self, device):
        src = torch.tensor([3, 1, -2], device=device)
        result = scatter(src, torch.tensor([0, 0, 1], device=device), reduce="min", dim_size=3)
        torch.testing.assert_close(result, src.new_tensor([1, -2, 0]))

    def test_empty_input(self, device):
        result = scatter(torch.empty((0, 2), device=device), torch.empty(0, dtype=torch.long, device=device))
        torch.testing.assert_close(result, torch.empty((0, 2), device=device))

    @pytest.mark.parametrize(
        "index,error",
        [([0.0, 1.0], TypeError), ([-1, 0], ValueError), ([0, 2], ValueError)],
    )
    def test_invalid_indices(self, index, error):
        with pytest.raises(error):
            scatter(torch.ones(2), torch.tensor(index), dim_size=2)


class TestScatterMean:
    @pytest.mark.parametrize(
        "index,expected",
        [
            ([0, 0, 1, 1], [[2, 3, 0], [6, 7, 0]]),
            ([[0, 0, 1, 1]], [[2, 3, 0], [6, 7, 0]]),
            ([[0, 0, 1, 1], [1, 0, 1, 0]], [[2, 3, 0], [7.5, 5.5, 0]]),
        ],
    )
    def test_broadcast_index_and_negative_dim(self, device, index, expected):
        src = torch.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 6.0, 8.0]], device=device)
        result = scatter_mean(src, torch.tensor(index, device=device), dim=-1, dim_size=3)
        torch.testing.assert_close(result, src.new_tensor(expected))

    @pytest.mark.parametrize(
        "dtype,value",
        [
            (torch.int8, 100),
            (torch.int64, torch.iinfo(torch.int64).max),
            (torch.bool, True),
            (torch.float16, torch.finfo(torch.float16).max),
            (torch.complex64, 2 + 4j),
        ],
    )
    def test_mean_dtype_and_range(self, device, dtype, value):
        src = torch.full((2,), value, dtype=dtype, device=device)
        result = scatter_mean(src, torch.zeros(2, dtype=torch.long, device=device), dim_size=2)
        expected_dtype = dtype if src.is_floating_point() or src.is_complex() else torch.float32
        torch.testing.assert_close(result, torch.tensor([value, 0], dtype=expected_dtype, device=device))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_bucket_mean_and_gradient(self, device, dtype):
        size = 70_000
        src = torch.ones(size, dtype=dtype, device=device, requires_grad=True)
        result = scatter_mean(src, torch.zeros(size, dtype=torch.long, device=device))
        torch.testing.assert_close(result, src.new_tensor([1]))
        torch.testing.assert_close(torch.autograd.grad(result.sum(), src)[0], torch.full_like(src, 1 / size))

    def test_empty_integer_mean(self, device):
        result = scatter_mean(
            torch.empty((2, 0), dtype=torch.int8, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            dim=1,
            dim_size=3,
        )
        torch.testing.assert_close(result, torch.zeros((2, 3), device=device))

    def test_fullgraph_compilation(self):
        src = torch.full((2, 4), 100.0, dtype=torch.float16, requires_grad=True)
        compiled = torch.compile(scatter_mean, backend="aot_eager", fullgraph=True)
        result = compiled(src, torch.tensor([[0, 0, 1, 1]]), dim=-1, dim_size=3)
        torch.testing.assert_close(result, src.new_tensor([[100, 100, 0], [100, 100, 0]]))
        torch.testing.assert_close(torch.autograd.grad(result.sum(), src)[0], torch.full_like(src, 0.5))
