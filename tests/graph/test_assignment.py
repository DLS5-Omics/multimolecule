# MultiMolecule
# Copyright (C) 2024-Present MultiMolecule

from __future__ import annotations

import itertools
import math
from random import Random

import pytest
import torch

from multimolecule.graph import linear_sum_assignment

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


def _validate_assignment(costs, pairs):
    rows, columns = costs.shape[-2:]
    size = min(rows, columns)
    assert pairs.shape == costs.shape[:-2] + (size, 2)
    assert pairs.dtype == torch.long
    assert pairs.device == costs.device
    if pairs.numel() == 0:
        return
    for matrix, selected in zip(costs.reshape(-1, rows, columns), pairs.reshape(-1, size, 2)):
        row, column = selected.unbind(-1)
        assert torch.equal(row, row.sort().values)
        assert torch.unique(row).numel() == torch.unique(column).numel() == size
        assert ((row >= 0) & (row < rows)).all()
        assert ((column >= 0) & (column < columns)).all()
        assert torch.isfinite(matrix[row, column]).all()


def _brute_force_objective(matrix, maximize=False):
    rows, columns = len(matrix), len(matrix[0])
    scores = []
    if rows <= columns:
        assignments = (
            list(zip(range(rows), permutation)) for permutation in itertools.permutations(range(columns), rows)
        )
    else:
        assignments = (
            list(zip(permutation, range(columns))) for permutation in itertools.permutations(range(rows), columns)
        )
    for pairs in assignments:
        values = [matrix[row][column] for row, column in pairs]
        if all(math.isfinite(value) for value in values):
            scores.append(sum(values))
    if not scores:
        raise ValueError("No complete finite assignment")
    return (max if maximize else min)(scores)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4)])
def test_assignment_matches_independent_bruteforce(device, maximize, dtype, shape):
    random = Random(917 + shape[0] * 100 + shape[1])
    data = [
        [random.randint(-20, 20) if dtype == torch.int64 else random.uniform(-5, 5) for _ in range(shape[1])]
        for _ in range(shape[0])
    ]
    costs = torch.tensor(data, dtype=dtype, device=device)
    before = costs.clone()

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    actual = sum(costs[row, column].item() for row, column in pairs)
    expected = _brute_force_objective(data, maximize)
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert torch.equal(costs, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (12, 12)])
def test_batched_assignment_matches_scipy_objectives(device, maximize, shape):
    scipy = pytest.importorskip("scipy.optimize")
    generator = torch.Generator().manual_seed(1937 + shape[0])
    costs_cpu = torch.randn((2, 3, *shape), dtype=torch.float64, generator=generator)
    costs = costs_cpu.to(device)

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    size = min(shape)
    for matrix, selected in zip(costs_cpu.reshape(-1, *shape), pairs.cpu().reshape(-1, size, 2)):
        row, column = scipy.linear_sum_assignment(matrix.numpy(), maximize=maximize)
        expected = matrix[row, column].sum().item()
        actual = matrix[selected[:, 0], selected[:, 1]].sum().item()
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
def test_noncontiguous_tall_batch_with_gradients_matches_bruteforce(device, maximize):
    generator = torch.Generator().manual_seed(836)
    storage = torch.randn((2, 6, 8), dtype=torch.float64, generator=generator).to(device)
    costs = storage[:, ::2, ::2].transpose(-1, -2).requires_grad_()
    assert not costs.is_contiguous()
    before = costs.detach().clone()

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    for matrix, selected in zip(costs.detach(), pairs):
        expected = _brute_force_objective(matrix.tolist(), maximize)
        actual = matrix[selected[:, 0], selected[:, 1]].sum().item()
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert torch.equal(costs, before)
    assert costs.grad is None
    assert not pairs.requires_grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.uint8, torch.float16, torch.bfloat16, torch.float32])
def test_assignment_accepts_real_numeric_dtypes(device, maximize, dtype):
    costs = torch.tensor([[4, 1], [2, 5]], dtype=dtype, device=device)

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    assert pairs.tolist() == ([[0, 0], [1, 1]] if maximize else [[0, 1], [1, 0]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (2, 0, 3), (0, 3, 4), (2, 0, 3, 4)])
def test_empty_assignment_shapes_preserve_batch_shape_and_device(device, shape):
    costs = torch.empty(shape, dtype=torch.float64, device=device)

    pairs = linear_sum_assignment(costs)

    _validate_assignment(costs, pairs)
    assert pairs.numel() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("shape", [(3, 3), (2, 5), (5, 2)])
def test_tied_costs_require_a_complete_valid_assignment(device, maximize, shape):
    costs = torch.full(shape, -3.0, dtype=torch.float64, device=device)

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    assert costs[pairs[:, 0], pairs[:, 1]].sum().item() == -3 * min(shape)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
def test_assignment_preserves_integer_differences_above_float64_precision(device, maximize):
    base = 2**55
    costs = torch.tensor(
        [[[base + 1, base], [base, base]], [[-base - 1, -base], [-base, -base]]], dtype=torch.int64, device=device
    )

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    diagonal, off_diagonal = [[0, 0], [1, 1]], [[0, 1], [1, 0]]
    assert pairs.tolist() == ([diagonal, off_diagonal] if maximize else [off_diagonal, diagonal])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize(
    "data,expected",
    [
        ([[4, math.inf, 1], [math.inf, 2, math.inf]], [[0, 2], [1, 1]]),
        ([[math.inf, math.inf], [2, math.inf], [math.inf, 3]], [[1, 0], [2, 1]]),
        ([[math.inf, 2, math.inf], [math.inf, math.inf, 3]], [[0, 1], [1, 2]]),
        (
            [
                [math.inf, math.inf, math.inf, 1, math.inf],
                [2, math.inf, math.inf, math.inf, math.inf],
                [math.inf, math.inf, math.inf, math.inf, 3],
                [math.inf, 4, 5, math.inf, math.inf],
                [math.inf, 6, 4, math.inf, math.inf],
            ],
            [[0, 3], [1, 0], [2, 4], [3, 1], [4, 2]],
        ),
    ],
)
def test_forbidden_costs_and_anchored_rows_can_still_be_feasible(device, maximize, data, expected):
    costs = torch.tensor(data, dtype=torch.float64, device=device)
    if maximize:
        costs = -costs

    pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)
    assert pairs.tolist() == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_hall_infeasibility_is_detected_even_when_every_row_and_column_has_a_finite_cost(device, maximize, batched):
    costs = torch.tensor([[0, math.inf, math.inf], [0, math.inf, math.inf], [math.inf, 0, 0]], device=device)
    if batched:
        costs = torch.stack((torch.zeros_like(costs), costs))
    if maximize:
        costs = -costs

    with pytest.raises(ValueError, match="infeasible|finite|assignment"):
        linear_sum_assignment(costs, maximize=maximize)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
def test_assignment_rejects_nan_and_wrong_sign_infinity(device, maximize):
    wrong_infinity = math.inf if maximize else -math.inf
    for value in (math.nan, wrong_infinity):
        costs = torch.tensor([[0.0, value], [1.0, 2.0]], device=device)
        with pytest.raises(ValueError):
            linear_sum_assignment(costs, maximize=maximize)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max, 10**308])
def test_assignment_rejects_values_outside_exact_arithmetic_range(device, value):
    dtype = torch.float64 if value == 10**308 else torch.int64
    costs = torch.tensor([[value, 0], [0, 0]], dtype=dtype, device=device)

    with pytest.raises(ValueError, match="range|large|overflow"):
        linear_sum_assignment(costs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
def test_assignment_does_not_export_costs_to_a_cpu_backend(monkeypatch, device, maximize):
    costs = torch.tensor([[[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], [[2.0, 6.0, 5.0], [3.0, 5.0, 8.0]]], device=device)

    def reject_export(*args, **kwargs):
        raise AssertionError("Assignment must not export matrix values or use a CPU library fallback")

    with monkeypatch.context() as patch:
        for method in ("tolist", "cpu", "numpy"):
            patch.setattr(torch.Tensor, method, reject_export)
        pairs = linear_sum_assignment(costs, maximize=maximize)

    _validate_assignment(costs, pairs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_assignment_has_a_fixed_number_of_host_scalar_checks():
    counts = []
    for size in (4, 16):
        costs = torch.arange(size * size, device="cuda", dtype=torch.float64).reshape(size, size)
        costs = ((costs * 17 + 5) % 37) - 18
        linear_sum_assignment(costs)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            pairs = linear_sum_assignment(costs)
            torch.cuda.synchronize()
        counts.append(sum(event.count for event in prof.key_averages() if event.key == "aten::_local_scalar_dense"))
        _validate_assignment(costs, pairs)
    assert counts[0] == counts[1], f"Host scalar reads must not grow with solver iterations: {counts}"
    assert counts[0] <= 8


def test_assignment_rejects_non_matrix_or_non_tensor_inputs():
    with pytest.raises(TypeError):
        linear_sum_assignment([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        linear_sum_assignment(torch.tensor(1.0))
    with pytest.raises(ValueError):
        linear_sum_assignment(torch.ones(3))
    with pytest.raises(TypeError):
        linear_sum_assignment(torch.ones((2, 2), dtype=torch.complex64))
    with pytest.raises(TypeError):
        linear_sum_assignment(torch.ones((2, 2), dtype=torch.bool))
    with pytest.raises(TypeError):
        linear_sum_assignment(torch.ones((2, 2)), maximize=1)
