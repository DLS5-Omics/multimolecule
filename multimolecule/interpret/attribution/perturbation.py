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

"""Attribution by perturbing input features."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from ..utils import OutputSelector, TensorOrTuple
from .adapter import (
    AttributionInputs,
    Baselines,
    InputKeys,
    Target,
    _captum,
    _forward,
    _prepare_inputs,
)
from .output import AttributionResult


def feature_ablation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    feature_mask: TensorOrTuple | None = None,
    perturbations_per_eval: int = 1,
    show_progress: bool = False,
) -> AttributionResult:
    """Measure score changes after replacing features with their baseline values.

    Supply ``feature_mask`` to replace grouped features together. This method supports
    discrete inputs and works without gradients.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().FeatureAblation(_forward(forward, output))
    return AttributionResult(
        attributor.attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
    )


def occlusion(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    sliding_window_shapes: tuple[int, ...] | tuple[tuple[int, ...], ...],
    input_keys: InputKeys | None = None,
    strides: int | tuple[int | tuple[int, ...], ...] | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    perturbations_per_eval: int = 1,
    show_progress: bool = False,
) -> AttributionResult:
    """Measure score changes after replacing sliding input windows with baseline values.

    Specify window dimensions without the batch axis, with one shape per input tensor.
    A scalar per example uses the empty shape. Positions covered by multiple windows
    receive the average score change over those windows.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    if not sliding_window_shapes:
        # Captum indexes the first window shape before normalizing a single input.
        sliding_window_shapes = ((),)
    attributor = _captum().Occlusion(_forward(forward, output))
    return AttributionResult(
        attributor.attribute(
            inputs,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
    )


def shapley_value_sampling(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    feature_mask: TensorOrTuple | None = None,
    n_samples: int = 25,
    perturbations_per_eval: int = 1,
    show_progress: bool = False,
) -> AttributionResult:
    """Estimate Shapley values from score changes across sampled feature orderings.

    Starting from ``baselines``, this method restores each feature in a sampled ordering.
    Supply ``feature_mask`` to restore groups of features together. Each position in a
    group receives that group's attribution. ``n_samples`` controls the number of
    sampled orderings.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().ShapleyValueSampling(_forward(forward, output))
    return AttributionResult(
        attributor.attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
    )


def shapley_values(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    feature_mask: TensorOrTuple | None = None,
    perturbations_per_eval: int = 1,
    show_progress: bool = False,
) -> AttributionResult:
    """Compute exact Shapley values by enumerating all feature orderings.

    The cost grows factorially with the number of feature groups. Use a grouped
    ``feature_mask`` for a small number of groups or use ``shapley_value_sampling``
    for larger inputs.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().ShapleyValues(_forward(forward, output))
    return AttributionResult(
        attributor.attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
    )


def feature_permutation(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    input_keys: InputKeys | None = None,
    feature_mask: TensorOrTuple | None = None,
    perturbations_per_eval: int = 1,
    show_progress: bool = False,
    perm_func: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> AttributionResult:
    """Attribute score differences after permuting feature values across examples.

    Each attributed input tensor needs at least two aligned examples. For mapping inputs,
    use ``input_keys`` to select the tensors to permute. A ``feature_mask`` groups
    features whose score change is measured together. A custom ``perm_func`` receives
    an input batch and a Boolean feature mask and returns the permuted input batch.
    """
    raw_forward, tensor_inputs, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    tensors = (tensor_inputs,) if isinstance(tensor_inputs, Tensor) else tensor_inputs
    if any(tensor.ndim == 0 or tensor.size(0) < 2 for tensor in tensors):
        raise ValueError("Feature permutation requires at least two examples in each input tensor.")

    make_attributor = _captum().FeaturePermutation
    forward = _forward(raw_forward, output)
    attributor = make_attributor(forward) if perm_func is None else make_attributor(forward, perm_func=perm_func)
    return AttributionResult(
        attributor.attribute(
            tensor_inputs,
            target=target,
            additional_forward_args=forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
    )
