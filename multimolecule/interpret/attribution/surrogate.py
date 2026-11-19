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

"""Attribution with local surrogate models."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol

from torch import Tensor, nn
from torch.utils.data import DataLoader

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


class _SurrogateModel(Protocol):
    def fit(self, train_data: DataLoader, **kwargs: Any) -> object: ...

    def representation(self) -> Tensor: ...


def _require_sklearn(method: str) -> None:
    try:
        import_module("sklearn")
    except ModuleNotFoundError as error:
        if error.name != "sklearn":
            raise
        raise ImportError(f"{method} requires scikit-learn. Install 'multimolecule[interpret]'.") from error


def kernel_shap(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    baselines: Baselines | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    input_keys: InputKeys | None = None,
    output: OutputSelector | None = None,
    feature_mask: TensorOrTuple | None = None,
    n_samples: int = 25,
    perturbations_per_eval: int = 1,
    return_input_shape: bool = True,
    show_progress: bool = False,
) -> AttributionResult:
    """Estimate Shapley values from a weighted linear model fitted to perturbed predictions.

    Supply ``feature_mask`` to group features. Captum requires at least two groups.
    By default, the result broadcasts each group's coefficient over the input shape.
    Set ``return_input_shape=False`` to get one coefficient per interpretable feature.
    """
    captum = _captum()
    _require_sklearn("Kernel SHAP")

    forward, tensor_inputs, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = captum.KernelShap(_forward(forward, output))
    return AttributionResult(
        attributor.attribute(
            tensor_inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )
    )


def lime(
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
    return_input_shape: bool = True,
    show_progress: bool = False,
    interpretable_model: _SurrogateModel | None = None,
    similarity_func: Callable[..., float | Tensor] | None = None,
    perturb_func: Callable[..., object] | None = None,
) -> AttributionResult:
    """Fit a local surrogate to predictions with input features switched on and off.

    A ``feature_mask`` ties groups of features together. Captum's default surrogate
    is a scikit-learn Lasso model. Pass an interpretable model to choose a different
    regression implementation.
    """
    captum = _captum()
    if interpretable_model is None:
        _require_sklearn("LIME")
    forward, tensor_inputs, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = captum.Lime(
        _forward(forward, output),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
        perturb_func=perturb_func,
    )
    return AttributionResult(
        attributor.attribute(
            tensor_inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )
    )


def lime_base(
    model: nn.Module,
    inputs: AttributionInputs,
    *,
    interpretable_model: _SurrogateModel,
    similarity_func: Callable[..., float | Tensor],
    perturb_func: Callable[..., object],
    perturb_interpretable_space: bool,
    from_interp_rep_transform: Callable[..., TensorOrTuple] | None = None,
    to_interp_rep_transform: Callable[..., Tensor] | None = None,
    input_keys: InputKeys | None = None,
    target: Target | None = None,
    additional_forward_args: object = None,
    output: OutputSelector | None = None,
    n_samples: int = 50,
    perturbations_per_eval: int = 1,
    show_progress: bool = False,
    **kwargs: Any,
) -> AttributionResult:
    """Fit a caller-supplied surrogate from custom perturbations and similarity weights.

    Provide the transform that maps the chosen sampling space into the other space.
    For a keyword batch, the callbacks receive the selected tensors in
    ``input_keys`` order. Extra keyword arguments are forwarded to Captum's callbacks.
    The attribution contains the fitted surrogate's representation.
    """
    forward, tensor_inputs, forward_args = _prepare_inputs(model, inputs, additional_forward_args, input_keys)
    attributor = _captum().LimeBase(
        _forward(forward, output),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
        perturb_func=perturb_func,
        perturb_interpretable_space=perturb_interpretable_space,
        from_interp_rep_transform=from_interp_rep_transform,
        to_interp_rep_transform=to_interp_rep_transform,
    )
    return AttributionResult(
        attributor.attribute(
            tensor_inputs,
            target=target,
            additional_forward_args=forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
            **kwargs,
        )
    )
