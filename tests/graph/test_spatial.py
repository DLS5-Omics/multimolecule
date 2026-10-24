# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import pytest
import torch

from multimolecule.graph import knn_graph, radius, radius_graph


def _assert_edges(actual, expected):
    assert actual.dtype == torch.long and actual.device == expected.device
    assert sorted(actual.T.tolist()) == sorted(expected.T.tolist())


def _radius_reference(query, reference, cutoff, limit, exclude_diagonal=False):
    distance = (query.double()[:, None] - reference.double()[None]).norm(dim=-1)
    if exclude_diagonal:
        distance.fill_diagonal_(float("inf"))
    valid = distance < cutoff
    if limit < len(reference):
        selected = distance.topk(limit, largest=False).indices
        valid &= torch.zeros_like(valid).scatter_(1, selected, True)
    return torch.stack(valid.nonzero(as_tuple=True))


class TestSpatialGraphs:
    def test_direction_and_nearest_neighbor_limit(self, device):
        pos = torch.tensor([[0.0], [1.0], [1.1]], device=device)
        expected = torch.tensor([[1, 2, 1], [0, 1, 2]], device=device)
        _assert_edges(radius_graph(pos, 2.0, max_num_neighbors=1), expected)
        _assert_edges(knn_graph(pos, 1), expected)
        x, y = pos[:2], torch.tensor([[0.5], [2.0], [5.0]], device=device)
        _assert_edges(radius(x, y, 1.1), torch.tensor([[0, 0, 1], [0, 1, 1]], device=device))

    def test_unsorted_batch_boundaries(self, device):
        pos = torch.tensor([[0.0], [0.5], [0.0], [0.5]], device=device)
        batch = torch.tensor([9, -2, 9, -2], dtype=torch.int32, device=device)
        expected = torch.tensor([[3, 1, 2, 0], [1, 3, 0, 2]], device=device)
        _assert_edges(radius_graph(pos, float("inf"), batch=batch), expected)
        _assert_edges(knn_graph(pos, 3, batch=batch), expected)

    @pytest.mark.parametrize("loop", [False, True])
    def test_strict_cutoff_loops_and_oversized_k(self, device, loop):
        pos = torch.tensor([[0.0], [1.0]], device=device)
        expected = (
            torch.arange(2, device=device).repeat(2, 1)
            if loop
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )
        _assert_edges(radius_graph(pos, 1.0, loop=loop), expected)
        edges = knn_graph(pos, 5, loop=loop)
        assert edges.shape == (2, 4 if loop else 2)
        assert (edges[0] == edges[1]).sum() == (2 if loop else 0)

    def test_empty_neighborhoods(self, device):
        pos = torch.tensor([[0.0], [1.0]], device=device)
        for result in (
            knn_graph(pos[:0], 1),
            knn_graph(pos[:1], 1),
            knn_graph(pos, 0),
            radius_graph(pos, 0.0),
            radius_graph(pos, 1.0, max_num_neighbors=0),
            radius(pos[:0], pos, 1.0),
        ):
            _assert_edges(result, torch.empty((2, 0), dtype=torch.long, device=device))

    def test_cross_radius_matches_only_shared_batches(self, device):
        x = torch.zeros((3, 1), device=device)
        y = x + 0.2
        batch_x = torch.tensor([9, -2, 77], device=device)
        batch_y = torch.tensor([9, -2, 88], device=device)
        _assert_edges(radius(x, y, 1.0, batch_x, batch_y), torch.tensor([[0, 1], [0, 1]], device=device))

    def test_sparse_cloud_against_dense_geometry(self, device):
        pos = torch.rand((1024, 3), generator=torch.Generator().manual_seed(389), dtype=torch.float64).to(device)
        expected = _radius_reference(pos, pos, 0.12, 3, exclude_diagonal=True).flip(0)
        _assert_edges(radius_graph(pos, 0.12, max_num_neighbors=3), expected)

    def test_downstream_position_gradients(self, device):
        pos = torch.tensor([[0.0], [1.0], [3.0]], device=device, requires_grad=True)
        source, target = knn_graph(pos, 1)
        (pos[source] - pos[target]).square().sum().backward()
        torch.testing.assert_close(pos.grad, pos.new_tensor([[-4], [0], [4]]))


class TestDistanceNumerics:
    @pytest.mark.parametrize(
        "pos,cutoff",
        [
            (1e6 + torch.arange(31, dtype=torch.float32)[:, None] * 0.5, 0.75),
            (torch.cat((torch.tensor([[1e6]]), torch.arange(31)[:, None] * 0.5)), 0.75),
            (torch.tensor([[-3e38], [-1.1e38], [0.0], [0.6e38], [3e38]]), 1.5e38),
            (1e150 + torch.arange(31, dtype=torch.float64)[:, None] * 1e135, 1.4e135),
        ],
    )
    def test_large_coordinates_preserve_neighbors(self, device, pos, cutoff):
        pos = pos.to(device)
        expected = _radius_reference(pos, pos, cutoff, 3, exclude_diagonal=True).flip(0)
        _assert_edges(radius_graph(pos, cutoff, max_num_neighbors=3), expected)
        source, target = knn_graph(pos, 1)
        distance = (pos.double()[:, None] - pos.double()[None]).norm(dim=-1)
        distance.fill_diagonal_(float("inf"))
        torch.testing.assert_close((pos[source].double() - pos[target].double()).norm(dim=-1), distance.amin(dim=1))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_distances(self, device, dtype):
        pos = torch.tensor([[-300.0], [300.0]], dtype=dtype, device=device)
        expected = torch.tensor([[1, 0], [0, 1]], device=device)
        _assert_edges(knn_graph(pos, 1), expected)
        _assert_edges(radius_graph(pos, 601.0), expected)
        _assert_edges(radius(pos[:1], pos[1:], 601.0), expected.new_tensor([[0], [0]]))


class TestSpatialValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"r": -1.0},
            {"r": 1.0, "pos": torch.tensor([[float("nan")], [0.0]])},
            {"r": 1.0, "pos": torch.tensor([[-1e300], [1e300]], dtype=torch.float64)},
        ],
    )
    def test_invalid_geometry_raises(self, kwargs):
        arguments = {"pos": torch.zeros((2, 1)), **kwargs}
        with pytest.raises(ValueError):
            radius_graph(**arguments)

    def test_cross_input_compatibility(self):
        pos = torch.zeros((2, 3))
        with pytest.raises(ValueError):
            radius(pos, pos[:, :2], 1.0)
