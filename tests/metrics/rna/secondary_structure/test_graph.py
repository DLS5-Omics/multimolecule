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
from types import SimpleNamespace

import pytest
import torch

from multimolecule.metrics.rna.secondary_structure import graph
from multimolecule.utils.rna.secondary_structure import LoopType, PseudoknotType


@dataclass(frozen=True)
class _LoopNode:
    kind: LoopType = LoopType.HAIRPIN
    size: int = 3
    branch_count: int = 0
    asymmetry: int = 0
    pseudoknot_type: PseudoknotType = PseudoknotType.NONE
    role: str | None = None


def _context(
    *, pred_loops=None, target_loops=None, pred_helices=None, target_helices=None, pred_edges=None, target_edges=None
):
    return SimpleNamespace(
        device=torch.device("cpu"),
        loop_helix_assignment=(
            [] if pred_loops is None else pred_loops,
            [] if target_loops is None else target_loops,
            [] if pred_helices is None else pred_helices,
            [] if target_helices is None else target_helices,
            [] if pred_edges is None else pred_edges,
            [] if target_edges is None else target_edges,
            {},
            {},
        ),
    )


class TestLoopHelixGraphGed:
    def test_identical_loop_helix_graph_has_zero_distance(self) -> None:
        context = _context(
            pred_loops=[_LoopNode()],
            target_loops=[_LoopNode()],
            pred_helices=[((0, 5), (1, 4))],
            target_helices=[((0, 5), (1, 4))],
            pred_edges=[(0, 0)],
            target_edges=[(0, 0)],
        )

        assert graph.loop_helix_graph_ged(context) == pytest.approx(0.0)

    def test_edge_changes_are_counted_after_helix_node_offset(self) -> None:
        context = _context(
            pred_loops=[_LoopNode()],
            target_loops=[_LoopNode()],
            pred_helices=[((0, 5),)],
            target_helices=[((0, 5),)],
            pred_edges=[],
            target_edges=[(0, 0)],
        )

        assert graph.loop_helix_graph_ged(context) == pytest.approx(0.2)

    def test_loop_and_helix_node_substitution_costs_are_used(self) -> None:
        loop_mismatch = _context(
            pred_loops=[_LoopNode(kind=LoopType.HAIRPIN)],
            target_loops=[_LoopNode(kind=LoopType.INTERNAL)],
            pred_helices=[((0, 5),)],
            target_helices=[((0, 5),)],
            pred_edges=[(0, 0)],
            target_edges=[(0, 0)],
        )
        helix_length_mismatch = _context(
            pred_loops=[_LoopNode()],
            target_loops=[_LoopNode()],
            pred_helices=[((0, 5),)],
            target_helices=[((0, 5), (1, 4))],
            pred_edges=[(0, 0)],
            target_edges=[(0, 0)],
        )

        assert graph.loop_helix_graph_ged(loop_mismatch) > 0.0
        assert graph.loop_helix_graph_ged(helix_length_mismatch) > 0.0

    def test_empty_to_nonempty_graph_has_full_distance(self) -> None:
        context = _context(
            target_loops=[_LoopNode()],
            target_helices=[((0, 5),)],
            target_edges=[(0, 0)],
        )

        assert graph.loop_helix_graph_ged(context) == pytest.approx(1.0)
