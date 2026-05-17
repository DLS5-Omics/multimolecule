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

from typing import Literal

from torch import Tensor

from ..outputs import AttributionOutput
from ..targets import ScalarTarget

AttributionMethod = Literal[
    "feature_ablation",
    "gradient_shap",
    "input_x_gradient",
    "integrated_gradients",
    "layer_integrated_gradients",
    "occlusion",
    "saliency",
]
Baseline = Literal["pad", "mask", "zero"] | Tensor
AttributionTarget = ScalarTarget

__all__ = ["AttributionMethod", "AttributionOutput", "AttributionTarget", "Baseline"]
