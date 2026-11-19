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

"""Model-agnostic feature attribution with Captum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from torch import Tensor, nn

from .._model import LayerRef, OutputSelector, resolve_layer, select_tensor

if TYPE_CHECKING:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients

__all__ = ["AttributionResult", "feature_ablation", "integrated_gradients", "layer_integrated_gradients"]

TensorOrTuple: TypeAlias = Tensor | tuple[Tensor, ...]
Baselines: TypeAlias = Tensor | int | float | tuple[Tensor | int | float, ...]
Target: TypeAlias = int | tuple[int, ...] | list[int] | list[tuple[int, ...]] | Tensor


@dataclass(frozen=True)
class AttributionResult:
    """Captum attributions and optional integration residuals."""

    attributions: TensorOrTuple | list[TensorOrTuple]
    """Attributions in Captum's original shape and structure."""

    convergence_delta: Tensor | None = None
    """Per-example convergence delta when requested for an integrated gradients algorithm."""


def integrated_gradients(
    model: nn.Module,
    inputs: TensorOrTuple,
    *,
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

    The model receives ``inputs`` and ``additional_forward_args`` positionally. The returned
    tensor or the ``logits`` field provides scores by default. Use a field name or callable
    as ``output`` to select another tensor, and ``target`` to select a score per example.
    """
    attributor = _captum().IntegratedGradients(_forward(model, output), multiply_by_inputs=multiply_by_inputs)
    return _gradient_result(
        attributor,
        inputs,
        return_convergence_delta,
        baselines=baselines,
        target=target,
        additional_forward_args=additional_forward_args,
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
    )


def layer_integrated_gradients(
    model: nn.Module,
    inputs: TensorOrTuple,
    *,
    layer: LayerRef | list[LayerRef],
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
    captum = _captum()
    module = (
        [resolve_layer(model, item)[1] for item in layer] if isinstance(layer, list) else resolve_layer(model, layer)[1]
    )
    if isinstance(module, list) and len({id(item) for item in module}) != len(module):
        raise ValueError("Each layer must refer to a different module.")
    device_ids = model.device_ids if isinstance(model, nn.DataParallel) else None
    attributor = captum.LayerIntegratedGradients(
        _forward(model, output), module, device_ids=device_ids or None, multiply_by_inputs=multiply_by_inputs
    )
    return _gradient_result(
        attributor,
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


def feature_ablation(
    model: nn.Module,
    inputs: TensorOrTuple,
    *,
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
    attributor = _captum().FeatureAblation(_forward(model, output))
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


def _captum():
    try:
        from captum import attr
    except ModuleNotFoundError as error:
        if error.name != "captum":
            raise
        raise ImportError("Attribution requires Captum. Install 'multimolecule[interpret]'.") from error
    return attr


def _forward(model: nn.Module, output: OutputSelector | None):
    def forward(*args: Any) -> Tensor:
        return select_tensor(model(*args), output)

    return forward


def _gradient_result(
    attributor: IntegratedGradients | LayerIntegratedGradients,
    inputs: TensorOrTuple,
    return_convergence_delta: bool,
    **kwargs: Any,
) -> AttributionResult:
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Run gradient attribution outside torch.inference_mode().")
    if return_convergence_delta:
        attributions, delta = attributor.attribute(inputs, return_convergence_delta=True, **kwargs)
        return AttributionResult(attributions, delta)
    return AttributionResult(attributor.attribute(inputs, **kwargs))
