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

"""Input-gradient attribution methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from torch import nn

from ..utils import OutputSelector, TensorOrTuple
from .adapter import (
    AttributionInputs,
    Baselines,
    InputKeys,
    Target,
    _captum,
    _forward,
    _gradient_result,
    _prepare_inputs,
)
from .output import AttributionResult


def integrated_gradients(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_steps: int = 50,
    method: str = "gausslegendre",
    internal_batch_size: int | None = None,
    return_convergence_delta: bool = False,
    multiply_by_inputs: bool = True,
) -> AttributionResult:
    """Attribute continuous inputs by integrating gradients from a baseline.

    Tensor inputs and ``additional_forward_args`` are positional. With a mapping, select
    the attributed fields using ``input_keys`` and pass all other model arguments as fields
    in that mapping. The returned tensor or ``logits`` supplies the scores by default.
    Use ``output`` to select another tensor and ``target`` to select a score per example.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().IntegratedGradients(_forward(forward, output), multiply_by_inputs=multiply_by_inputs)
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
    )


def saliency(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    abs: bool = True,
) -> AttributionResult:
    """Compute the gradient of the selected score with respect to each input.

    Returns gradient magnitudes by default. Pass ``abs=False`` to retain the sign.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().Saliency(_forward(forward, output))
    return _gradient_result(
        attributor.attribute, inputs, target=target, additional_forward_args=additional_forward_args, abs=abs
    )


def input_x_gradient(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
) -> AttributionResult:
    """Multiply each input feature by its signed gradient for the selected score."""
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().InputXGradient(_forward(forward, output))
    return _gradient_result(
        attributor.attribute, inputs, target=target, additional_forward_args=additional_forward_args
    )


def gradient_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    baselines: TensorOrTuple | Callable[..., TensorOrTuple],
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_samples: int = 5,
    stdevs: float | tuple[float, ...] = 0.0,
    return_convergence_delta: bool = False,
    multiply_by_inputs: bool = True,
) -> AttributionResult:
    """Estimate gradient attributions by sampling from a baseline distribution.

    Provide a batch of baseline examples per input tensor or a callable that supplies
    those batches. The optional convergence delta contains one value per sampled baseline
    and input perturbation, ordered by input example.
    """
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().GradientShap(_forward(forward, output), multiply_by_inputs=multiply_by_inputs)
    return _gradient_result(
        attributor.attribute,
        inputs,
        return_convergence_delta=return_convergence_delta,
        baselines=baselines,
        target=target,
        additional_forward_args=additional_forward_args,
        n_samples=n_samples,
        stdevs=stdevs,
    )


def smoothgrad(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    nt_samples: int = 5,
    nt_samples_batch_size: int | None = None,
    stdevs: float | tuple[float, ...] = 1.0,
    abs: bool = True,
) -> AttributionResult:
    """Average saliency over Gaussian perturbations of each input.

    Pass ``abs=False`` to average signed gradients instead of gradient magnitudes.
    """
    return _noisy_saliency(
        model,
        inputs,
        "smoothgrad",
        input_keys=input_keys,
        target=target,
        additional_forward_args=additional_forward_args,
        output=output,
        nt_samples=nt_samples,
        nt_samples_batch_size=nt_samples_batch_size,
        stdevs=stdevs,
        abs=abs,
    )


def smoothgrad_sq(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    nt_samples: int = 5,
    nt_samples_batch_size: int | None = None,
    stdevs: float | tuple[float, ...] = 1.0,
    abs: bool = True,
) -> AttributionResult:
    """Average squared saliency over Gaussian perturbations of each input."""
    return _noisy_saliency(
        model,
        inputs,
        "smoothgrad_sq",
        input_keys=input_keys,
        target=target,
        additional_forward_args=additional_forward_args,
        output=output,
        nt_samples=nt_samples,
        nt_samples_batch_size=nt_samples_batch_size,
        stdevs=stdevs,
        abs=abs,
    )


def vargrad(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    nt_samples: int = 5,
    nt_samples_batch_size: int | None = None,
    stdevs: float | tuple[float, ...] = 1.0,
    abs: bool = True,
) -> AttributionResult:
    """Measure the variance of saliency across Gaussian perturbations."""
    return _noisy_saliency(
        model,
        inputs,
        "vargrad",
        input_keys=input_keys,
        target=target,
        additional_forward_args=additional_forward_args,
        output=output,
        nt_samples=nt_samples,
        nt_samples_batch_size=nt_samples_batch_size,
        stdevs=stdevs,
        abs=abs,
    )


def _noisy_saliency(
    model: nn.Module,
    inputs: AttributionInputs,
    nt_type: Literal["smoothgrad", "smoothgrad_sq", "vargrad"],
    *,
    input_keys: InputKeys | None,
    target: Target | None,
    additional_forward_args: object,
    output: OutputSelector | None,
    nt_samples: int,
    nt_samples_batch_size: int | None,
    stdevs: float | tuple[float, ...],
    abs: bool,
) -> AttributionResult:
    forward, inputs, additional_forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    captum = _captum()
    attributor = captum.NoiseTunnel(captum.Saliency(_forward(forward, output)))
    return _gradient_result(
        attributor.attribute,
        inputs,
        target=target,
        additional_forward_args=additional_forward_args,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=nt_samples_batch_size,
        stdevs=stdevs,
        abs=abs,
    )
