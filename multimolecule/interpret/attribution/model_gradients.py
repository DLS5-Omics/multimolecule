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

"""Attribution methods with model-wide propagation rules."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from ..utils import LayerRef, OutputSelector, TensorOrTuple, resolve_layer
from .adapter import (
    AttributionInputs,
    Baselines,
    InputKeys,
    Target,
    _captum,
    _device_ids,
    _gradient_result,
    _isolated_relu_inputs,
    _lrp_forward,
    _lrp_state,
    _module_forward,
    _prepare_inputs,
    _preserve_module_attributes,
)
from .output import AttributionResult


def deep_lift(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    return_convergence_delta: bool = False,
    custom_attribution_func: Callable[..., tuple[Tensor, ...]] | None = None,
    multiply_by_inputs: bool = True,
    eps: float = 1e-10,
) -> AttributionResult:
    """Attribute changes from a baseline with DeepLIFT's Rescale rule.

    Captum installs hooks on supported activation modules. Use distinct ``nn.Module``
    instances for non-linearities and use the model exclusively during attribution.
    Input and baseline pairs require a single model replica with ``nn.DataParallel``.
    """
    module, inputs, forward_args = _prepare_module(
        model, inputs, additional_forward_args, input_keys, output, paired_inputs=True
    )
    attributor = _captum().DeepLift(module, multiply_by_inputs=multiply_by_inputs, eps=eps)
    with _preserve_module_attributes(model, "input", "output", "input_ref"):
        return _gradient_result(
            attributor.attribute,
            inputs,
            return_convergence_delta=return_convergence_delta,
            baselines=baselines,
            target=target,
            additional_forward_args=forward_args,
            custom_attribution_func=custom_attribution_func,
        )


def deep_lift_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    baselines: TensorOrTuple | Callable[..., TensorOrTuple],
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    return_convergence_delta: bool = False,
    custom_attribution_func: Callable[..., tuple[Tensor, ...]] | None = None,
    multiply_by_inputs: bool = True,
) -> AttributionResult:
    """Average DeepLIFT attributions across a distribution of baselines.

    Supply multiple baseline examples per input tensor. The optional delta contains one
    entry per input and baseline pair. DeepLIFT's activation module and single-replica
    requirements apply.
    """
    module, inputs, forward_args = _prepare_module(
        model, inputs, additional_forward_args, input_keys, output, paired_inputs=True
    )
    attributor = _captum().DeepLiftShap(module, multiply_by_inputs=multiply_by_inputs)
    with _preserve_module_attributes(model, "input", "output", "input_ref"):
        return _gradient_result(
            attributor.attribute,
            inputs,
            return_convergence_delta=return_convergence_delta,
            baselines=baselines,
            target=target,
            additional_forward_args=forward_args,
            custom_attribution_func=custom_attribution_func,
        )


def guided_backprop(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
) -> AttributionResult:
    """Restrict input gradients to positive signal through active ReLU units.

    Captum applies the rule to ``nn.ReLU`` modules and hooks the model during attribution.
    """
    module, inputs, forward_args = _prepare_module(model, inputs, additional_forward_args, input_keys, output)
    attributor = _captum().GuidedBackprop(module)
    with _isolated_relu_inputs(model):
        return _gradient_result(attributor.attribute, inputs, target=target, additional_forward_args=forward_args)


def deconvolution(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
) -> AttributionResult:
    """Attribute inputs using positive backward signal at ``nn.ReLU`` modules.

    Captum propagates positive gradients through hooked ReLUs independently of their
    forward activations. Use the model exclusively during attribution.
    """
    module, inputs, forward_args = _prepare_module(model, inputs, additional_forward_args, input_keys, output)
    attributor = _captum().Deconvolution(module)
    with _isolated_relu_inputs(model):
        return _gradient_result(attributor.attribute, inputs, target=target, additional_forward_args=forward_args)


def guided_grad_cam(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    interpolate_mode: str = "nearest",
    attribute_to_layer_input: bool = False,
) -> AttributionResult:
    """Multiply guided input gradients by the resized positive Grad-CAM map.

    Use a spatial layer with one tensor output and inputs with matching spatial geometry.
    Guided Backprop's requirement for ``nn.ReLU`` modules applies. Captum temporarily
    attaches hooks to the model and layer during attribution.
    """
    module, inputs, forward_args = _prepare_module(model, inputs, additional_forward_args, input_keys, output)
    target_layer = resolve_layer(model, layer)[1]
    attributor = _captum().GuidedGradCam(module, target_layer, device_ids=_device_ids(model))
    with _isolated_relu_inputs(model):
        return _gradient_result(
            attributor.attribute,
            inputs,
            target=target,
            additional_forward_args=forward_args,
            interpolate_mode=interpolate_mode,
            attribute_to_layer_input=attribute_to_layer_input,
        )


def lrp(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    return_convergence_delta: bool = False,
    verbose: bool = False,
) -> AttributionResult:
    """Propagate a selected model score backward using layer relevance rules.

    Each leaf layer needs a built-in Captum rule or an explicit ``PropagationRule``.
    Captum temporarily hooks layers and may modify their weights during propagation.
    Use the model exclusively during attribution.
    """
    forward, inputs, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    module = _lrp_forward(model, forward, forward_args, output)
    with _lrp_state(model):
        attributor = _captum().LRP(module)
        return _gradient_result(
            attributor.attribute,
            inputs,
            return_convergence_delta=return_convergence_delta,
            target=target,
            verbose=verbose,
        )


def _prepare_module(
    model: nn.Module,
    inputs: AttributionInputs,
    additional_forward_args: object,
    input_keys: InputKeys | None,
    output: OutputSelector | None,
    *,
    paired_inputs: bool = False,
) -> tuple[nn.Module, TensorOrTuple, object]:
    forward, tensor_inputs, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    return _module_forward(model, forward, output, paired_inputs=paired_inputs), tensor_inputs, forward_args
