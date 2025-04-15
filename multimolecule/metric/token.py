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


from warnings import warn

import torch
from danling import NestedTensor
from danling.metric import AverageMeter, MetricMeter, MetricMeters
from torch import Tensor


class TokenMetric(MetricMeter):

    def update(
        self,
        input: Tensor | NestedTensor,
        target: Tensor | NestedTensor,
    ) -> None:
        if input.ndim < 3:
            warn(f"TokenMetric is designed for token/contact_map prediction. Input has {input.ndim} dimensions.")
        if target.ndim < 3:
            warn(f"TokenMetric is designed for token/contact_map prediction. Target has {target.ndim} dimensions.")
        inputs, targets = input.unbind(), target.unbind()
        if len(inputs) != len(targets):
            raise ValueError(
                f"Input and target must have the same number of elements. Got {len(inputs)} and {len(targets)}."
            )
        ret = torch.mean(torch.stack([self.metric(i, t) for i, t in zip(inputs, targets)]))
        AverageMeter.update(self, ret, len(inputs))


class TokenMetrics(MetricMeters):
    default_cls = TokenMetric
