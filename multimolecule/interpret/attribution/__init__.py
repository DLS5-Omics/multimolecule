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


from .attribute import attribute
from .base import ModelAttributor
from .feature_ablation import FeatureAblationAttributor
from .gradient_shap import GradientShapAttributor
from .input_x_gradient import InputXGradientAttributor
from .integrated_gradients import IntegratedGradientsAttributor
from .layer_integrated_gradients import LayerIntegratedGradientsAttributor
from .occlusion import OcclusionAttributor
from .registry import ATTRIBUTORS
from .saliency import SaliencyAttributor
from .types import AttributionMethod, AttributionOutput, AttributionTarget, Baseline

__all__ = [
    "ATTRIBUTORS",
    "AttributionMethod",
    "AttributionOutput",
    "AttributionTarget",
    "Baseline",
    "FeatureAblationAttributor",
    "GradientShapAttributor",
    "IntegratedGradientsAttributor",
    "InputXGradientAttributor",
    "LayerIntegratedGradientsAttributor",
    "ModelAttributor",
    "OcclusionAttributor",
    "SaliencyAttributor",
    "attribute",
]
