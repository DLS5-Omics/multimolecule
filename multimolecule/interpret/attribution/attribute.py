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

from typing import Any

from torch import Tensor, nn

from .registry import ATTRIBUTORS
from .types import AttributionMethod, AttributionOutput, AttributionTarget, Baseline


def attribute(
    model: nn.Module,
    input_ids: Tensor,
    target: AttributionTarget | int | None = None,
    *,
    method: AttributionMethod = "layer_integrated_gradients",
    baseline: Baseline = "pad",
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    **kwargs: Any,
) -> AttributionOutput:
    attributor = ATTRIBUTORS.build(method, model)
    return attributor(
        input_ids=input_ids,
        target=target,
        baseline=baseline,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **kwargs,
    )


__all__ = ["attribute"]
