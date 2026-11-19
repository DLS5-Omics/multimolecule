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

"""Feature attribution for PyTorch models using Captum.

Pass tensors positionally, or pass a keyword batch and use ``input_keys`` to
select the tensor fields to attribute. Remaining tensor fields with the same
leading batch size expand with the attributed fields. Broadcast and other fixed
arguments retain their original values and shapes.
For multiple attributed fields, pass per-input baselines and masks in
``input_keys`` order. Methods returning per-input tuples use that same order.

Check the method docstring for layer and operator requirements. DeepLIFT and
LRP rely on compatible modules, and Grad-CAM uses convolutional feature maps.
"""

from .input_gradients import (
    gradient_shap,
    input_x_gradient,
    integrated_gradients,
    saliency,
    smoothgrad,
    smoothgrad_sq,
    vargrad,
)
from .layer import (
    internal_influence,
    layer_activation,
    layer_conductance,
    layer_deep_lift,
    layer_deep_lift_shap,
    layer_feature_ablation,
    layer_feature_permutation,
    layer_grad_cam,
    layer_gradient_shap,
    layer_gradient_x_activation,
    layer_integrated_gradients,
    layer_lrp,
)
from .model_gradients import (
    deconvolution,
    deep_lift,
    deep_lift_shap,
    guided_backprop,
    guided_grad_cam,
    lrp,
)
from .neuron import (
    neuron_conductance,
    neuron_deconvolution,
    neuron_deep_lift,
    neuron_deep_lift_shap,
    neuron_feature_ablation,
    neuron_gradient,
    neuron_gradient_shap,
    neuron_guided_backprop,
    neuron_integrated_gradients,
)
from .output import AttributionResult
from .perturbation import (
    feature_ablation,
    feature_permutation,
    occlusion,
    shapley_value_sampling,
    shapley_values,
)
from .surrogate import (
    kernel_shap,
    lime,
    lime_base,
)

__all__ = [
    "AttributionResult",
    "deconvolution",
    "deep_lift",
    "deep_lift_shap",
    "feature_ablation",
    "feature_permutation",
    "gradient_shap",
    "guided_backprop",
    "guided_grad_cam",
    "input_x_gradient",
    "integrated_gradients",
    "internal_influence",
    "kernel_shap",
    "layer_activation",
    "layer_conductance",
    "layer_deep_lift",
    "layer_deep_lift_shap",
    "layer_feature_ablation",
    "layer_feature_permutation",
    "layer_grad_cam",
    "layer_gradient_shap",
    "layer_gradient_x_activation",
    "layer_integrated_gradients",
    "layer_lrp",
    "lime",
    "lime_base",
    "lrp",
    "neuron_conductance",
    "neuron_deconvolution",
    "neuron_deep_lift",
    "neuron_deep_lift_shap",
    "neuron_feature_ablation",
    "neuron_gradient",
    "neuron_gradient_shap",
    "neuron_guided_backprop",
    "neuron_integrated_gradients",
    "occlusion",
    "saliency",
    "shapley_value_sampling",
    "shapley_values",
    "smoothgrad",
    "smoothgrad_sq",
    "vargrad",
]
