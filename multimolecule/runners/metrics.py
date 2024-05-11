# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from chanfig import Registry as Registry_
from danling.metrics import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics


class Registry(Registry_):

    def build(self, type, num_labels: int | None = None, **kwargs):
        if type == "multilabel":
            return self.init(self.lookup(type), num_labels=num_labels, **kwargs)
        if type == "multiclass":
            return self.init(self.lookup(type), num_classes=num_labels, **kwargs)
        if type == "regression":
            return self.init(self.lookup(type), num_outputs=num_labels, **kwargs)
        return self.init(self.lookup(type), **kwargs)


MetricRegistry = Registry(key="type")
MetricRegistry.register(binary_metrics, "binary")
MetricRegistry.register(multiclass_metrics, "multiclass")
MetricRegistry.register(multilabel_metrics, "multilabel")
MetricRegistry.register(regression_metrics, "regression")
