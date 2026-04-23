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

import inspect

import torch

import multimolecule.metrics.rna.secondary_structure as ss_metrics
from multimolecule.metrics.rna.secondary_structure import RnaSecondaryStructureContext


def _public_metric_functions():
    for name in ss_metrics.__all__:
        item = getattr(ss_metrics, name)
        if inspect.isfunction(item):
            yield name, item


def test_public_exports_resolve() -> None:
    assert ss_metrics.__all__
    for name in ss_metrics.__all__:
        assert getattr(ss_metrics, name) is not None


def test_public_metric_exports_execute(pr_positive_context: RnaSecondaryStructureContext) -> None:
    for name, metric_fn in _public_metric_functions():
        output = metric_fn(pr_positive_context)

        if name.endswith("_precision_recall_curve"):
            assert isinstance(output, tuple)
            assert len(output) == 3
            precision, recall, thresholds = output
            assert isinstance(precision, torch.Tensor)
            assert isinstance(recall, torch.Tensor)
            assert isinstance(thresholds, torch.Tensor)
            assert precision.ndim == recall.ndim == thresholds.ndim == 1
        elif name.endswith("_confusion"):
            assert isinstance(output, torch.Tensor)
            assert output.shape == (2, 2)
        else:
            assert isinstance(output, torch.Tensor)
            assert output.ndim == 0
