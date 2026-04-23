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

import torch

from multimolecule.metrics.rna.secondary_structure import topology
from multimolecule.utils.rna.secondary_structure import StemSegment
from tests.metrics.rna.secondary_structure.utils import assert_context_confusion_metric_family


@dataclass
class _FakeTopology:
    segments: list[StemSegment]
    edges: list[tuple[int, int, int]]

    def stem_graph_components(self) -> tuple[list[StemSegment], list[tuple[int, int, int]]]:
        return self.segments, self.edges


@dataclass
class _FakeContext:
    length: int
    device: torch.device
    pred_topology: _FakeTopology
    target_topology: _FakeTopology


def test_stem_assignment_and_edges_empty_graphs() -> None:
    ctx = _FakeContext(
        length=0,
        device=torch.device("cpu"),
        pred_topology=_FakeTopology([], []),
        target_topology=_FakeTopology([], []),
    )
    assert topology._stem_assignment_and_edges(ctx) == (0, 0, 0, 0, 0, 0)


def test_stem_assignment_and_edges_counts_matches_and_dedupes_edges() -> None:
    pred_segments = [StemSegment(0, 1, 9, 8, tier=0), StemSegment(3, 5, 7, 5, tier=0)]
    tgt_segments = [StemSegment(0, 1, 9, 8, tier=0), StemSegment(3, 5, 7, 5, tier=0)]
    pred_edges = [(0, 1, 1), (0, 1, 1), (1, 0, 2)]
    tgt_edges = [(0, 1, 1), (1, 0, 2), (1, 0, 2)]

    ctx = _FakeContext(
        length=10,
        device=torch.device("cpu"),
        pred_topology=_FakeTopology(pred_segments, pred_edges),
        target_topology=_FakeTopology(tgt_segments, tgt_edges),
    )

    matched_nodes, pred_nodes, tgt_nodes, matched_edges, pred_edge_count, tgt_edge_count = (
        topology._stem_assignment_and_edges(ctx)
    )
    assert matched_nodes == 2
    assert pred_nodes == 2
    assert tgt_nodes == 2
    assert matched_edges == 2
    assert pred_edge_count == 2
    assert tgt_edge_count == 2


def test_topology_metric_family_uses_context_confusion(perfect_nested_context) -> None:
    assert_context_confusion_metric_family(
        perfect_nested_context,
        perfect_nested_context.topology_confusion,
        topology.topology_precision,
        topology.topology_recall,
        topology.topology_f1,
    )
