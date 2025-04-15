# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from chanfig import Registry as Registry_
from danling.metric import Metrics

from multimolecule.tasks import Task

from .factory import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics
from .token import TokenMetrics


class Registry(Registry_):
    case_sensitive = False

    def build(
        self,
        task: Task,
        **kwargs,
    ) -> Metrics | TokenMetrics:
        cls = Metrics if task.level == "sequence" else TokenMetrics
        if task.type == "multilabel":
            return self.lookup(task.type)(cls=cls, num_labels=task.num_labels, **kwargs)
        if task.type == "multiclass":
            return self.lookup(task.type)(cls=cls, num_classes=task.num_labels, **kwargs)
        if task.type == "regression":
            return self.lookup(task.type)(cls=cls, num_outputs=task.num_labels, **kwargs)
        return self.lookup(task.type)(cls=cls, **kwargs)


METRICS = Registry(key="task")
METRICS.register(binary_metrics, "binary")
METRICS.register(multiclass_metrics, "multiclass")
METRICS.register(multilabel_metrics, "multilabel")
METRICS.register(regression_metrics, "regression")
