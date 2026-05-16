# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import pytest
import torch

from multimolecule.graph import knn_graph, radius, radius_graph, scatter, scatter_mean


def test_scatter_reductions() -> None:
    src = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 0, 1])

    assert scatter(src, index, dim_size=2, reduce="sum").tolist() == [3.0, 3.0]
    assert scatter_mean(src, index, dim_size=2).tolist() == [1.5, 3.0]
    assert scatter(src, index, dim_size=2, reduce="max").tolist() == [2.0, 3.0]


def test_scatter_handles_empty_and_integer_min_max() -> None:
    assert scatter(torch.tensor([]), torch.tensor([], dtype=torch.long)).shape == (0,)
    assert scatter(torch.empty((0, 2)), torch.tensor([], dtype=torch.long), dim=0).shape == (0, 2)

    src = torch.tensor([1, 2, 3])
    index = torch.tensor([0, 0, 1])

    assert scatter(src, index, dim_size=3, reduce="max").tolist() == [2, 3, 0]
    assert scatter(src, index, dim_size=3, reduce="min").tolist() == [1, 3, 0]


def test_scatter_supports_negative_dim_and_full_shaped_index() -> None:
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert scatter(src, torch.tensor([0, 1]), dim=-1, dim_size=3).tolist() == [
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
    ]
    assert scatter_mean(src, torch.tensor([[0, 1], [1, 0]]), dim=1, dim_size=2).tolist() == [
        [1.0, 2.0],
        [4.0, 3.0],
    ]


def test_scatter_rejects_invalid_indices() -> None:
    with pytest.raises(TypeError, match="integer"):
        scatter(torch.tensor([1.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        scatter(torch.tensor([1.0]), torch.tensor([-1]))
    with pytest.raises(ValueError, match="dim_size"):
        scatter(torch.tensor([1.0]), torch.tensor([1]), dim_size=1)


def test_spatial_graph_builders() -> None:
    pos = torch.tensor([[0.0], [1.0], [3.0]])

    assert radius_graph(pos, r=1.1, loop=False).tolist() == [[1, 0], [0, 1]]
    assert knn_graph(pos, k=1, loop=False).tolist() == [[1, 0, 1], [0, 1, 2]]

    x = torch.tensor([[0.0], [2.0]])
    y = torch.tensor([[0.5], [3.0]])
    assert radius(x, y, r=1.1).tolist() == [[0, 1], [0, 1]]


def test_spatial_graph_builders_respect_batch_boundaries() -> None:
    pos = torch.tensor([[0.0], [0.5], [0.0], [0.5]])
    batch = torch.tensor([0, 0, 1, 1])

    assert radius_graph(pos, r=1.0, batch=batch, loop=False).tolist() == [[1, 0, 3, 2], [0, 1, 2, 3]]
    assert knn_graph(pos, k=1, batch=batch, loop=False).tolist() == [[1, 0, 3, 2], [0, 1, 2, 3]]


def test_radius_graph_limits_nearest_neighbors_per_target() -> None:
    pos = torch.tensor([[0.0], [1.0], [1.1]])

    assert radius_graph(pos, r=2.0, max_num_neighbors=1, loop=False).tolist() == [[1, 2, 1], [0, 1, 2]]


def test_spatial_graph_builders_validate_inputs() -> None:
    pos = torch.tensor([[0.0], [1.0]])

    with pytest.raises(ValueError, match="k"):
        knn_graph(pos, k=-1)
    with pytest.raises(ValueError, match="max_num_neighbors"):
        radius_graph(pos, r=1.0, max_num_neighbors=-1)
    with pytest.raises(ValueError, match="batch"):
        radius_graph(pos, r=1.0, batch=torch.tensor([0]))
    with pytest.raises(ValueError, match="finite"):
        radius_graph(torch.tensor([[0.0], [float("nan")]]), r=1.0)
    with pytest.raises(ValueError, match="finite"):
        knn_graph(torch.tensor([[0.0], [float("inf")]]), k=1)
