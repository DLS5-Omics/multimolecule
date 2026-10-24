# MultiMolecule
# Copyright (C) 2024-Present MultiMolecule

import pytest
import torch
from scipy.optimize import linear_sum_assignment as scipy_assignment

from multimolecule.graph import linear_sum_assignment


@pytest.fixture(params=[(3, 5), (5, 3), (4, 4)], ids=["wide", "tall", "square"])
def costs(request, device):
    rows, columns = request.param
    generator = torch.Generator().manual_seed(1937)
    storage = torch.randn((2, 2, rows, 2 * columns), generator=generator, dtype=torch.float64)
    if rows == columns:
        storage = storage[0, 0]
    return storage.to(device)[..., ::2].requires_grad_()


class TestLinearSumAssignment:
    @pytest.mark.parametrize("maximize", [False, True])
    def test_batched_objective_matches_scipy(self, costs, maximize):
        before = costs.detach().clone()

        pairs = linear_sum_assignment(costs, maximize=maximize)

        rows, columns = costs.shape[-2:]
        size = min(rows, columns)
        assert pairs.shape == (*costs.shape[:-2], size, 2)
        assert pairs.dtype == torch.long and pairs.device == costs.device
        for matrix, selected in zip(before.reshape(-1, rows, columns), pairs.reshape(-1, size, 2)):
            row, column = selected.unbind(-1)
            assert torch.equal(row, row.sort().values)
            assert row.unique().numel() == column.unique().numel() == size
            assert ((0 <= row) & (row < rows)).all()
            assert ((0 <= column) & (column < columns)).all()
            expected_row, expected_column = scipy_assignment(matrix.cpu().numpy(), maximize=maximize)
            expected = matrix[expected_row, expected_column].sum().item()
            assert matrix[row, column].sum().item() == pytest.approx(expected, abs=1e-12)
        assert torch.equal(costs, before)
        assert costs.grad is None and not pairs.requires_grad

    @pytest.mark.parametrize("maximize", [False, True])
    def test_forbidden_edges_still_allow_complete_assignment(self, device, maximize):
        costs = torch.full((8, 10), torch.inf, dtype=torch.float64, device=device)
        costs[0, :8] = torch.arange(8, device=device)
        for row in range(1, 8):
            costs[row, row - 1] = -row
        if maximize:
            costs = -costs

        pairs = linear_sum_assignment(costs, maximize=maximize)

        assert pairs.shape == (8, 2)
        assert pairs[:, 0].unique().numel() == pairs[:, 1].unique().numel() == 8
        assert costs[pairs[:, 0], pairs[:, 1]].sum().item() == (21 if maximize else -21)

    @pytest.mark.parametrize("maximize", [False, True])
    def test_infeasible_batch_raises(self, device, maximize):
        infeasible = torch.tensor(
            [[0, torch.inf, torch.inf], [0, torch.inf, torch.inf], [torch.inf, 0, 0]], device=device
        )
        costs = torch.stack((torch.zeros_like(infeasible), infeasible))
        with pytest.raises(ValueError, match="infeasible"):
            linear_sum_assignment(-costs if maximize else costs, maximize=maximize)

    @pytest.mark.parametrize("maximize", [False, True])
    def test_integer_differences_above_float64_precision(self, device, maximize):
        base = 2**55
        costs = torch.tensor(
            [[[base + 1, base], [base, base]], [[-base - 1, -base], [-base, -base]]], dtype=torch.int64, device=device
        )

        pairs = linear_sum_assignment(costs, maximize=maximize)

        diagonal, off_diagonal = [[0, 0], [1, 1]], [[0, 1], [1, 0]]
        assert pairs.tolist() == ([diagonal, off_diagonal] if maximize else [off_diagonal, diagonal])

    def test_ties_return_valid_optimum(self, device):
        costs = torch.full((3, 5), -3.0, device=device)

        pairs = linear_sum_assignment(costs)

        assert pairs.shape == (3, 2)
        assert pairs[:, 0].unique().numel() == pairs[:, 1].unique().numel() == 3
        assert costs[pairs[:, 0], pairs[:, 1]].sum().item() == -9

    @pytest.mark.parametrize("shape", [(2, 0, 3), (0, 3, 4)])
    def test_empty_dimensions(self, device, shape):
        pairs = linear_sum_assignment(torch.empty(shape, device=device))

        assert pairs.shape == (*shape[:-2], min(shape[-2:]), 2)
        assert pairs.dtype == torch.long and pairs.device == device

    @pytest.mark.parametrize("maximize,value", [(False, torch.nan), (False, -torch.inf), (True, torch.inf)])
    def test_invalid_cost_values_raise(self, maximize, value):
        with pytest.raises(ValueError):
            linear_sum_assignment(torch.tensor([[0.0, value], [1.0, 2.0]]), maximize=maximize)
