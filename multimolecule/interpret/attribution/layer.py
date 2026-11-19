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

"""Attribution of intermediate layer features."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import torch
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
    _layer_modules,
    _lrp_forward,
    _lrp_state,
    _module_forward,
    _prepare_inputs,
    _preserve_module_attributes,
)
from .output import AttributionResult

_BaselineDistribution: TypeAlias = TensorOrTuple | Callable[..., TensorOrTuple]
_CustomAttribution: TypeAlias = Callable[..., tuple[Tensor, ...]]


def layer_integrated_gradients(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef | list[LayerRef],
    input_keys: InputKeys | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_steps: int = 50,
    method: str = "gausslegendre",
    internal_batch_size: int | None = None,
    return_convergence_delta: bool = False,
    attribute_to_layer_input: bool = False,
    multiply_by_inputs: bool = True,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Attribute an intermediate layer using Integrated Gradients.

    Pass a module, a dotted module path, or a list of independent layers that each run
    once per forward. By default the scores describe the layer output. Set
    ``attribute_to_layer_input`` to use its input.
    Use the hooked layers exclusively while Captum's forward hooks are active.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LayerIntegratedGradients(
        _forward(forward, output),
        _layer_modules(model, layer),
        device_ids=_device_ids(model),
        multiply_by_inputs=multiply_by_inputs,
    )
    return _gradient_result(
        attributor.attribute,
        inputs,
        return_convergence_delta,
        baselines=baselines,
        target=target,
        additional_forward_args=additional_forward_args,
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
        attribute_to_layer_input=attribute_to_layer_input,
        grad_kwargs=grad_kwargs,
    )


def layer_gradient_x_activation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef | list[LayerRef],
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    attribute_to_layer_input: bool = False,
    multiply_by_inputs: bool = True,
    memory_efficient: bool = False,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Multiply the score gradient with respect to a layer by its activation.

    Use ``attribute_to_layer_input`` to select the layer input or
    ``multiply_by_inputs=False`` to return the unweighted gradients. Pass a module,
    a dotted module path, or a list of distinct modules that each run once per
    forward. Use the selected layers exclusively while Captum's hooks are active.
    If a layer shares the selected tensor with a bypass, the tensor gradient
    includes the bypass contribution.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LayerGradientXActivation(
        _forward(forward, output),
        _layer_modules(model, layer),
        device_ids=_device_ids(model),
        multiply_by_inputs=multiply_by_inputs,
        memory_efficient=memory_efficient,
    )
    return _gradient_result(
        attributor.attribute,
        inputs,
        target=target,
        additional_forward_args=additional_forward_args,
        attribute_to_layer_input=attribute_to_layer_input,
        grad_kwargs=grad_kwargs,
    )


def layer_feature_ablation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    input_keys: InputKeys | None = None,
    layer_baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    layer_mask: TensorOrTuple | None = None,
    attribute_to_layer_input: bool = False,
    perturbations_per_eval: int = 1,
) -> AttributionResult:
    """Measure score changes after replacing selected layer features.

    ``layer_baselines`` supplies replacement values for the layer activations
    and defaults to zero. Use ``attribute_to_layer_input`` to ablate its inputs.
    ``layer_mask`` groups features by integer ID.
    Each feature in a group receives the score change for the entire group.
    The selected layer should run once per forward. Use it exclusively while
    Captum's hooks are active.

    With Captum 0.9, use ``perturbations_per_eval=1`` for per-example targets.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    module = resolve_layer(model, layer)[1]
    attributor = _captum().LayerFeatureAblation(_forward(forward, output), module, device_ids=_device_ids(model))
    return AttributionResult(
        attributor.attribute(
            inputs,
            layer_baselines=layer_baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            layer_mask=layer_mask,
            attribute_to_layer_input=attribute_to_layer_input,
            perturbations_per_eval=perturbations_per_eval,
        )
    )


def layer_activation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef | list[LayerRef],
    input_keys: InputKeys | None = None,
    additional_forward_args: object = None,
    attribute_to_layer_input: bool = False,
) -> AttributionResult:
    """Collect raw Captum activations from selected modules during one forward.

    Choose a module or dotted path. For a list of layers, results follow that
    list's order. Set ``attribute_to_layer_input`` to collect layer inputs.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LayerActivation(forward, _layer_modules(model, layer), device_ids=_device_ids(model))
    return AttributionResult(
        attributor.attribute(
            tensors, additional_forward_args=forward_args, attribute_to_layer_input=attribute_to_layer_input
        )
    )


def layer_conductance(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    baselines: Baselines | None = None,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_steps: int = 50,
    method: str = "gausslegendre",
    internal_batch_size: int | None = None,
    return_convergence_delta: bool = False,
    attribute_to_layer_input: bool = False,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Integrate layer contributions along the path from baseline to continuous inputs."""
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LayerConductance(
        _forward(forward, output), resolve_layer(model, layer)[1], device_ids=_device_ids(model)
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        return_convergence_delta=return_convergence_delta,
        baselines=baselines,
        target=target,
        additional_forward_args=forward_args,
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
        attribute_to_layer_input=attribute_to_layer_input,
        grad_kwargs=grad_kwargs,
    )


def internal_influence(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    baselines: Baselines | None = None,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_steps: int = 50,
    method: str = "gausslegendre",
    internal_batch_size: int | None = None,
    attribute_to_layer_input: bool = False,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Integrate the score gradient with respect to a layer along a continuous input path."""
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().InternalInfluence(
        _forward(forward, output), resolve_layer(model, layer)[1], device_ids=_device_ids(model)
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        baselines=baselines,
        target=target,
        additional_forward_args=forward_args,
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
        attribute_to_layer_input=attribute_to_layer_input,
        grad_kwargs=grad_kwargs,
    )


def layer_grad_cam(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    attribute_to_layer_input: bool = False,
    relu_attributions: bool = False,
    attr_dim_summation: bool = True,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Compute a Grad-CAM map from a layer with channels on axis 1.

    Gradients are averaged across spatial dimensions. By default the weighted
    activations are summed over channels and retain their sign.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LayerGradCam(
        _forward(forward, output), resolve_layer(model, layer)[1], device_ids=_device_ids(model)
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        target=target,
        additional_forward_args=forward_args,
        attribute_to_layer_input=attribute_to_layer_input,
        relu_attributions=relu_attributions,
        attr_dim_summation=attr_dim_summation,
        grad_kwargs=grad_kwargs,
    )


def layer_deep_lift(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    baselines: Baselines | None = None,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    return_convergence_delta: bool = False,
    attribute_to_layer_input: bool = False,
    multiply_by_inputs: bool = True,
    custom_attribution_func: _CustomAttribution | None = None,
    grad_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Attribute a layer using DeepLIFT's rescale rule.

    Rescaling applies to Captum's supported nonlinear modules. Each hooked
    nonlinearity should run once per forward. Use one model replica to keep
    each input and its baseline together.
    """
    return _layer_deep_lift_result(
        _captum().LayerDeepLift,
        model,
        inputs,
        layer,
        input_keys,
        additional_forward_args,
        output,
        multiply_by_inputs,
        return_convergence_delta,
        baselines=baselines,
        target=target,
        attribute_to_layer_input=attribute_to_layer_input,
        custom_attribution_func=custom_attribution_func,
        grad_kwargs=grad_kwargs,
    )


def layer_deep_lift_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    baselines: _BaselineDistribution,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    return_convergence_delta: bool = False,
    attribute_to_layer_input: bool = False,
    multiply_by_inputs: bool = True,
    custom_attribution_func: _CustomAttribution | None = None,
) -> AttributionResult:
    """Average layer DeepLIFT attributions across a distribution of baselines.

    Use one model replica to keep each input and its baselines together.
    """
    return _layer_deep_lift_result(
        _captum().LayerDeepLiftShap,
        model,
        inputs,
        layer,
        input_keys,
        additional_forward_args,
        output,
        multiply_by_inputs,
        return_convergence_delta,
        baselines=baselines,
        target=target,
        attribute_to_layer_input=attribute_to_layer_input,
        custom_attribution_func=custom_attribution_func,
    )


def _layer_deep_lift_result(
    method: Callable[..., Any],
    model: nn.Module,
    inputs: AttributionInputs,
    layer: LayerRef,
    input_keys: InputKeys | None,
    additional_forward_args: object,
    output: OutputSelector | None,
    multiply_by_inputs: bool,
    return_convergence_delta: bool,
    **kwargs: Any,
) -> AttributionResult:
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    single = isinstance(tensors, Tensor)
    values = (tensors,) if isinstance(tensors, Tensor) else tensors
    # LayerDeepLift registers backward hooks even on nonlinearities before the selected layer.
    grad_inputs = tuple(
        value.detach().requires_grad_() if value.is_floating_point() and not value.requires_grad else value
        for value in values
    )
    wrapped = _module_forward(model, forward, output, paired_inputs=True)
    attributor = method(wrapped, resolve_layer(model, layer)[1], multiply_by_inputs=multiply_by_inputs)
    with _preserve_module_attributes(model, "input", "output", "input_ref"):
        return _gradient_result(
            attributor.attribute,
            grad_inputs[0] if single else grad_inputs,
            return_convergence_delta=return_convergence_delta,
            additional_forward_args=forward_args,
            **kwargs,
        )


def layer_gradient_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    baselines: _BaselineDistribution,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_samples: int = 5,
    stdevs: float | tuple[float, ...] = 0.0,
    return_convergence_delta: bool = False,
    attribute_to_layer_input: bool = False,
    multiply_by_inputs: bool = True,
) -> AttributionResult:
    """Estimate layer attributions by sampling noisy continuous inputs and baselines.

    The model inputs must accept interpolation and additive noise, including when
    ``stdevs=0``. Optional convergence deltas follow the individual samples.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LayerGradientShap(
        _forward(forward, output),
        resolve_layer(model, layer)[1],
        device_ids=_device_ids(model),
        multiply_by_inputs=multiply_by_inputs,
    )
    return _gradient_result(
        attributor.attribute,
        tensors,
        return_convergence_delta=return_convergence_delta,
        baselines=baselines,
        target=target,
        additional_forward_args=forward_args,
        n_samples=n_samples,
        stdevs=stdevs,
        attribute_to_layer_input=attribute_to_layer_input,
    )


def layer_feature_permutation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    layer_mask: TensorOrTuple | None = None,
    perturbations_per_eval: int = 1,
) -> AttributionResult:
    """Measure score changes after shuffling layer outputs across examples.

    Supply a batch of at least two examples. Use integer IDs in ``layer_mask``
    to shuffle groups together.
    """
    captum = _captum()
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = captum.LayerFeaturePermutation(
        _permutation_forward(_forward(forward, output), tensors, target),
        resolve_layer(model, layer)[1],
        device_ids=_device_ids(model),
    )
    return AttributionResult(
        attributor.attribute(
            tensors,
            additional_forward_args=forward_args,
            layer_mask=layer_mask,
            perturbations_per_eval=perturbations_per_eval,
        )
    )


def _permutation_forward(
    forward: Callable[..., Tensor], inputs: TensorOrTuple, target: Target | None
) -> Callable[..., Tensor]:
    if target is None:
        return forward

    # Captum 0.9 selects its target both inside and outside the layer hook.
    # Bind it to the forward function so each evaluation selects its score once.
    if isinstance(target, Tensor):
        if target.is_floating_point() or target.is_complex() or target.dtype == torch.bool:
            raise TypeError("Tensor targets must contain integer indices.")
        if target.numel() == 1:
            target = int(target.item())
        elif target.ndim != 1:
            raise ValueError("Per-example tensor targets must be one-dimensional.")

    if isinstance(target, (int, tuple)):
        columns = (target,) if isinstance(target, int) else target
        if any(not isinstance(index, int) for index in columns):
            raise TypeError("Target coordinates must contain integer indices.")

        def select_columns(*args: Any) -> Tensor:
            scores = forward(*args)
            if len(columns) >= scores.ndim:
                raise ValueError("Target coordinates must fit the model output dimensions after the batch axis.")
            return scores[(slice(None), *columns)]

        return select_columns

    batch = (inputs if isinstance(inputs, Tensor) else inputs[0]).size(0)
    if not batch or len(target) != batch:
        raise ValueError("Per-example targets must match the input batch size.")

    if isinstance(target, Tensor):
        indices = target
    else:
        first = target[0]
        if isinstance(first, tuple):
            if any(
                not isinstance(value, tuple)
                or len(value) != len(first)
                or any(not isinstance(index, int) for index in value)
                for value in target
            ):
                raise TypeError("Per-example target tuples must contain the same number of integer indices.")
        elif any(not isinstance(value, int) for value in target):
            raise TypeError("Per-example targets must contain integer indices or tuples of integer indices.")
        indices = torch.tensor(target, dtype=torch.long)

    def select(*args: Any) -> Tensor:
        scores = forward(*args)
        if not scores.ndim:
            raise ValueError("Per-example targets require a batch axis in the model output.")
        repeats, remainder = divmod(scores.size(0), batch)
        if remainder or not repeats:
            raise ValueError("Per-example targets require one model output per input example.")
        local_indices = indices.to(device=scores.device, dtype=torch.long)
        if local_indices.ndim == 1:
            if scores.ndim != 2:
                raise ValueError("Per-example class targets require model outputs with shape (batch, classes).")
            return scores.gather(1, local_indices.repeat(repeats).unsqueeze(1)).squeeze(1)
        if local_indices.size(1) >= scores.ndim:
            raise ValueError("Target coordinates must fit the model output dimensions after the batch axis.")
        rows = torch.arange(scores.size(0), device=scores.device)
        return scores[(rows, *local_indices.repeat(repeats, 1).unbind(1))]

    return select


def layer_lrp(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    layer: LayerRef | list[LayerRef],
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    return_convergence_delta: bool = False,
    attribute_to_layer_input: bool = False,
    verbose: bool = False,
) -> AttributionResult:
    """Propagate selected score relevance to one or more layers.

    Captum requires relevance propagation rules for the model's supported
    leaf operations. A list of layers yields a matching list of relevances and
    convergence deltas.
    """
    forward, tensors, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    wrapped = _lrp_forward(model, forward, forward_args, output)
    with _lrp_state(model):
        attributor = _captum().LayerLRP(wrapped, _layer_modules(model, layer))
        return _gradient_result(
            attributor.attribute,
            tensors,
            return_convergence_delta=return_convergence_delta,
            target=target,
            attribute_to_layer_input=attribute_to_layer_input,
            verbose=verbose,
        )
