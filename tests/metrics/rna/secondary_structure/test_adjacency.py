# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
#
# This file is part of MultiMolecule.
#
# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from multimolecule.metrics.rna.secondary_structure import adjacency
from multimolecule.utils.rna.secondary_structure import LoopSpan, LoopType
from tests.metrics.rna.secondary_structure.utils import assert_context_confusion_metric_family


def test_adjacency_score_matrix_branches() -> None:
    device = torch.device("cpu")
    opposite = torch.tensor(
        [
            [0.9, 0.4, 0.1],
            [0.2, 0.8, 0.6],
            [0.7, 0.3, 0.5],
        ],
        dtype=torch.float32,
    )
    pred_adj = [[], [0], [0, 1], [0, 1, 2]]
    tgt_adj = [[], [1], [0, 1], [0, 1, 2]]
    score = adjacency._adjacency_score_matrix(pred_adj, tgt_adj, opposite, device)

    assert score.shape == (4, 4)
    assert score[0, 0] == pytest.approx(1.0)  # both empty
    assert score[1, 0] == pytest.approx(0.0)  # one empty
    assert score[1, 1] == pytest.approx(0.4)  # 1x1
    assert score[1, 2] == pytest.approx(0.9)  # 1xN takes best single match
    assert score[2, 1] == pytest.approx(0.8)  # Nx1 takes best single match
    assert score[2, 2] == pytest.approx(0.85)  # 2x2 special branch
    assert 0.0 <= float(score[3, 3].item()) <= 1.0  # generic matching

    neg = torch.full((3, 3), -1.0, dtype=torch.float32)
    score_neg = adjacency._adjacency_score_matrix(pred_adj, tgt_adj, neg, device)
    assert score_neg[2, 2] == pytest.approx(0.0)
    assert score_neg[3, 3] == pytest.approx(0.0)


def test_joint_score_and_global_assignment() -> None:
    device = torch.device("cpu")
    base = torch.tensor([[0.8, -1.0], [0.2, 0.4]], dtype=torch.float32)
    pred_adj = [[0], [1]]
    tgt_adj = [[0], [1]]
    opposite = torch.tensor([[0.9, 0.0], [0.0, 0.7]], dtype=torch.float32)

    joint = adjacency._joint_node_score(base, pred_adj, tgt_adj, opposite, device)
    assert joint.shape == base.shape
    assert joint[0, 1] == pytest.approx(-1.0)
    assert 0.0 <= float(joint[0, 0].item()) <= 1.0

    empty_joint = adjacency._joint_node_score(torch.empty((0, 0)), [], [], torch.empty((0, 0)), device)
    assert empty_joint.shape == (0, 0)

    loop_score = torch.tensor([[0.9, -1.0], [-1.0, 0.8]], dtype=torch.float32)
    helix_score = torch.tensor([[0.7]], dtype=torch.float32)
    loop_map, helix_map = adjacency._global_node_assignment(loop_score, helix_score, device)
    assert loop_map == {0: 0, 1: 1}
    assert helix_map == {0: 0}

    loop_map_empty, helix_map_empty = adjacency._global_node_assignment(
        torch.empty((0, 0)),
        torch.empty((0, 0)),
        device,
    )
    assert loop_map_empty == {}
    assert helix_map_empty == {}


def test_count_consistent_edge_matches() -> None:
    pred_edges = [(0, 0), (1, 0), (2, 1)]
    tgt_edges = [(10, 20), (11, 20)]
    loop_map = {0: 10, 1: 11}
    helix_map = {0: 20}
    matches = adjacency._count_consistent_edge_matches(pred_edges, tgt_edges, loop_map, helix_map)
    assert matches == 2


@dataclass
class _FakeLoop:
    tag: str
    kind: LoopType
    spans: tuple[LoopSpan, ...]
    size: int


def test_loop_score_matrix_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cpu")
    pred_loops = [
        _FakeLoop("p0", LoopType.HAIRPIN, (LoopSpan(0, 2),), 3),
        _FakeLoop("p1", LoopType.HAIRPIN, (LoopSpan(4, 6),), 3),
    ]
    tgt_loops = [
        _FakeLoop("t0", LoopType.HAIRPIN, (LoopSpan(1, 3),), 3),
        _FakeLoop("t1", LoopType.HAIRPIN, (LoopSpan(4, 6),), 3),
    ]

    monkeypatch.setattr(
        adjacency,
        "loop_anchor_pairs_score_any",
        lambda pred, tgt: 0.0 if (pred.tag, tgt.tag) == ("p0", "t1") else 1.0,
    )
    monkeypatch.setattr(
        adjacency,
        "loop_anchor_wiring_score",
        lambda _kind, pred, tgt, *_args: 0.0 if (pred.tag, tgt.tag) == ("p1", "t0") else 1.0,
    )
    monkeypatch.setattr(
        adjacency,
        "loop_anchor_order_score_any",
        lambda pred, tgt, *_args: 0.0 if (pred.tag, tgt.tag) == ("p1", "t1") else 1.0,
    )
    monkeypatch.setattr(adjacency, "loop_anchor_adjacency_score", lambda *_args: 1.0)
    monkeypatch.setattr(adjacency, "loop_anchor_coaxial_score", lambda *_args: 1.0)

    score = adjacency._loop_score_matrix(
        pred_loops,
        tgt_loops,
        {},
        {},
        {},
        [],
        [],
        device,
    )

    assert score.shape == (2, 2)
    assert float(score[0, 0].item()) > 0.0
    assert float(score[0, 1].item()) == -1.0
    assert float(score[1, 0].item()) == -1.0
    assert float(score[1, 1].item()) == -1.0

    empty = adjacency._loop_score_matrix([], tgt_loops, {}, {}, {}, [], [], device)
    assert empty.shape == (0, 2)


def test_loop_helix_edge_metric_family_uses_context_confusion(perfect_nested_context) -> None:
    assert_context_confusion_metric_family(
        perfect_nested_context,
        perfect_nested_context.loop_helix_edges_confusion,
        adjacency.loop_helix_edges_precision,
        adjacency.loop_helix_edges_recall,
        adjacency.loop_helix_edges_f1,
    )
