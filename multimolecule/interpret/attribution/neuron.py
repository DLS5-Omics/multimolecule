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

"""Attribution with intermediate neuron targets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from torch import Tensor, nn

from ..utils import LayerRef, OutputSelector, TensorOrTuple, resolve_layer
from .adapter import (
    AttributionInputs,
    Baselines,
    InputKeys,
    Target,
    _captum,
    _device_ids,
    _forward,
    _gradient_result,
    _isolated_relu_inputs,
    _module_forward,
    _prepare_inputs,
    _preserve_module_attributes,
)
from .output import AttributionResult

NeuronSelector: TypeAlias = int | tuple[int | slice, ...] | Callable[[TensorOrTuple], Tensor]

BaselineDistribution: TypeAlias = TensorOrTuple | Callable[..., TensorOrTuple]
CustomAttribution: TypeAlias = Callable[..., tuple[Tensor, ...]]


def neuron_integrated_gradients(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    baselines: Baselines | None = None,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    n_steps: int = 50,
    method: str = "gausslegendre",
    internal_batch_size: int | None = None,
    attribute_to_neuron_input: bool = False,
    multiply_by_inputs: bool = True,
) -> AttributionResult:
    """Attribute continuous model inputs to a neuron in the selected layer.

    Select a neuron by index, a tuple of indices or slices, or a callable that
    returns a scalar or one score per example from the layer activation. Slices
    sum the selected neurons before attribution. Set ``attribute_to_neuron_input``
    to select from the layer's input.

    For a mapping of model inputs, ``input_keys`` names the tensors to attribute.
    The selected layer should run once per forward. Use it exclusively while
    Captum's hooks are active.
    """
    captum = _captum()
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = captum.NeuronIntegratedGradients(
        forward, module, device_ids=_device_ids(model), multiply_by_inputs=multiply_by_inputs
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        neuron_selector=neuron_selector,
        baselines=baselines,
        additional_forward_args=forward_args,
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
        attribute_to_neuron_input=attribute_to_neuron_input,
    )


def neuron_conductance(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    output: OutputSelector | None = None,
    n_steps: int = 50,
    method: str = "riemann_trapezoid",
    internal_batch_size: int | None = None,
    attribute_to_neuron_input: bool = False,
    multiply_by_inputs: bool = True,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Attribute a selected prediction to input features through a single neuron.

    Select the prediction tensor with ``output`` and its score with ``target``.
    ``neuron_selector`` selects the intermediate neuron used for the attribution.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronConductance(
        _forward(forward, output), module, device_ids=_device_ids(model), multiply_by_inputs=multiply_by_inputs
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        neuron_selector=neuron_selector,
        baselines=baselines,
        target=target,
        additional_forward_args=forward_args,
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
        attribute_to_neuron_input=attribute_to_neuron_input,
        grad_kwargs=grad_kwargs,
    )


def neuron_feature_ablation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    baselines: Baselines | None = None,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    feature_mask: TensorOrTuple | None = None,
    attribute_to_neuron_input: bool = False,
    perturbations_per_eval: int = 1,
) -> AttributionResult:
    """Measure the selected neuron's response to replacing features with their baselines.

    Use ``feature_mask`` to replace groups of input features together.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronFeatureAblation(forward, module, device_ids=_device_ids(model))
    return AttributionResult(
        attributor.attribute(
            tensors,
            neuron_selector=neuron_selector,
            baselines=baselines,
            additional_forward_args=forward_args,
            feature_mask=feature_mask,
            attribute_to_neuron_input=attribute_to_neuron_input,
            perturbations_per_eval=perturbations_per_eval,
        )
    )


def neuron_gradient(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    attribute_to_neuron_input: bool = False,
) -> AttributionResult:
    """Differentiate the selected neuron with respect to each attributed model input."""
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronGradient(forward, module, device_ids=_device_ids(model))
    return _gradient_result(
        attributor.attribute,
        tensors,
        neuron_selector=neuron_selector,
        additional_forward_args=forward_args,
        attribute_to_neuron_input=attribute_to_neuron_input,
    )


def neuron_deep_lift(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    baselines: Baselines | None = None,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    output: OutputSelector | None = None,
    attribute_to_neuron_input: bool = False,
    multiply_by_inputs: bool = True,
    custom_attribution_func: CustomAttribution | None = None,
) -> AttributionResult:
    """Attribute a neuron's change from its reference activation using DeepLIFT.

    DeepLIFT's hooks pair inputs with baselines on a single replica and require a
    batch-aligned output tensor. Use ``output`` to select it from structured results.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronDeepLift(
        _module_forward(model, forward, output, paired_inputs=True), module, multiply_by_inputs=multiply_by_inputs
    )
    with _preserve_module_attributes(model, "input", "output", "input_ref"):
        return _gradient_result(
            attributor.attribute,
            tensors,
            neuron_selector=neuron_selector,
            baselines=baselines,
            additional_forward_args=forward_args,
            attribute_to_neuron_input=attribute_to_neuron_input,
            custom_attribution_func=custom_attribution_func,
        )


def neuron_deep_lift_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    baselines: BaselineDistribution,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    output: OutputSelector | None = None,
    attribute_to_neuron_input: bool = False,
    multiply_by_inputs: bool = True,
    custom_attribution_func: CustomAttribution | None = None,
) -> AttributionResult:
    """Average a neuron's DeepLIFT attributions over a distribution of baselines.

    The baseline distribution must contain multiple examples. Input and baseline
    pairs run on a single replica. Use ``output`` to select a tensor from structured results.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronDeepLiftShap(
        _module_forward(model, forward, output, paired_inputs=True), module, multiply_by_inputs=multiply_by_inputs
    )
    with _preserve_module_attributes(model, "input", "output", "input_ref"):
        return _gradient_result(
            attributor.attribute,
            tensors,
            neuron_selector=neuron_selector,
            baselines=baselines,
            additional_forward_args=forward_args,
            attribute_to_neuron_input=attribute_to_neuron_input,
            custom_attribution_func=custom_attribution_func,
        )


def neuron_gradient_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    baselines: BaselineDistribution,
    n_samples: int = 5,
    stdevs: float | tuple[float, ...] = 0.0,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    attribute_to_neuron_input: bool = False,
    multiply_by_inputs: bool = True,
) -> AttributionResult:
    """Estimate a neuron's Gradient SHAP attribution from sampled reference paths."""
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronGradientShap(
        forward, module, device_ids=_device_ids(model), multiply_by_inputs=multiply_by_inputs
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        neuron_selector=neuron_selector,
        baselines=baselines,
        n_samples=n_samples,
        stdevs=stdevs,
        additional_forward_args=forward_args,
        attribute_to_neuron_input=attribute_to_neuron_input,
    )


def neuron_deconvolution(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    attribute_to_neuron_input: bool = False,
) -> AttributionResult:
    """Backpropagate from a neuron with nonnegative gradients at ReLU outputs.

    Use ``torch.nn.ReLU`` modules to expose the activation hooks.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronDeconvolution(
        _module_forward(model, forward, raw_output=True), module, device_ids=_device_ids(model)
    )
    with _isolated_relu_inputs(model):
        return _gradient_result(
            attributor.attribute,
            tensors,
            neuron_selector=neuron_selector,
            additional_forward_args=forward_args,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )


def neuron_guided_backprop(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    neuron_selector: NeuronSelector,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    attribute_to_neuron_input: bool = False,
) -> AttributionResult:
    """Backpropagate from a neuron through active ReLUs with nonnegative gradients.

    Use ``torch.nn.ReLU`` modules to expose the activation hooks.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    _, module = resolve_layer(model, layer)
    attributor = _captum().NeuronGuidedBackprop(
        _module_forward(model, forward, raw_output=True), module, device_ids=_device_ids(model)
    )
    with _isolated_relu_inputs(model):
        return _gradient_result(
            attributor.attribute,
            tensors,
            neuron_selector=neuron_selector,
            additional_forward_args=forward_args,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )
