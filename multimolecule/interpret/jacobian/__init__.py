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

"""Derivatives of a selected model output with respect to inputs or layer activations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn

from .._model import LayerRef, ModelInputs, OutputSelector, call_model, resolve_layer, select_tensor

__all__ = ["full", "jvp", "vjp"]

TensorInputs: TypeAlias = Tensor | tuple[Tensor, ...]


def jvp(
    model: nn.Module,
    inputs: TensorInputs,
    tangents: TensorInputs,
    *,
    output: OutputSelector | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the selected output and its directional derivative along the input tangents.

    Pass one tangent with the shape of each differentiable model input tensor.
    The model must run with operators that support forward-mode automatic differentiation.
    """
    forward = _selected_forward(model, output, model_kwargs)
    model_inputs = (inputs,) if isinstance(inputs, Tensor) else inputs
    input_tangents = (tangents,) if isinstance(tangents, Tensor) else tangents
    return torch.func.jvp(forward, model_inputs, input_tangents)


def vjp(
    model: nn.Module,
    inputs: ModelInputs,
    cotangent: Tensor,
    *,
    wrt: LayerRef | None = None,
    output: OutputSelector | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Tensor, TensorInputs]:
    """Return the selected output and its cotangent projected onto inputs or a layer.

    The cotangent has the shape of the selected output. Input projections follow the
    structure of ``inputs``. Pass a submodule or its path as ``wrt`` to project onto its
    activation. On models with ``get_input_embeddings()``, ``wrt="embedding"`` uses that
    module. The layer must run once. Its activation and the selected model output must
    be floating-point tensors. Use the model exclusively during the hooked forward.
    """
    if wrt is not None:
        value, activation = _layer_forward(model, inputs, wrt, output, model_kwargs)
        if cotangent.shape != value.shape:
            raise ValueError("The cotangent must have the same shape as the selected output.")
        return value, _layer_vjp(value, activation, cotangent)

    forward = _selected_forward(model, output, model_kwargs)
    model_inputs = _input_tensors(inputs)
    value, pullback = torch.func.vjp(forward, *model_inputs)
    projections = pullback(cotangent)
    return value, projections[0] if isinstance(inputs, Tensor) else projections


def full(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    wrt: LayerRef | None = None,
    output: OutputSelector | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    chunk_size: int | None = None,
) -> TensorInputs:
    """Compute the full Jacobian of the selected output with respect to inputs or a layer.

    Each result has the output dimensions followed by the corresponding input or layer
    dimensions, including any batch dimensions. Use ``chunk_size`` to limit the number
    of derivative rows computed together. A size of one computes each row individually.
    The returned tensor contains one derivative per pair of output and input elements.

    Pass a submodule or its path as ``wrt`` for layer derivatives. A single forward
    uses the supplied model inputs and captures its floating-point activation. Use the
    model exclusively during that forward.
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if wrt is not None:
        value, activation = _layer_forward(model, inputs, wrt, output, model_kwargs)
        return _layer_jacobian(value, activation, chunk_size)

    forward = _selected_forward(model, output, model_kwargs)
    if isinstance(inputs, Tensor):
        return torch.func.jacrev(forward, chunk_size=chunk_size)(inputs)
    model_inputs = _input_tensors(inputs)
    return torch.func.jacrev(forward, argnums=tuple(range(len(model_inputs))), chunk_size=chunk_size)(*model_inputs)


def _input_tensors(inputs: ModelInputs) -> tuple[Tensor, ...]:
    if isinstance(inputs, Tensor):
        return (inputs,)
    if isinstance(inputs, tuple) and inputs and all(isinstance(value, Tensor) for value in inputs):
        return inputs
    raise TypeError("Pass a tensor or tuple of tensors for input derivatives. Use wrt for layer derivatives.")


def _check_gradient_mode() -> None:
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Run differentiation outside torch.inference_mode().")


def _selected_forward(
    model: nn.Module,
    output: OutputSelector | None,
    model_kwargs: Mapping[str, Any] | None,
) -> Callable[..., Tensor]:
    _check_gradient_mode()

    def forward(*inputs: Tensor) -> Tensor:
        return select_tensor(call_model(model, inputs, model_kwargs), output)

    return forward


def _layer_forward(
    model: nn.Module,
    inputs: ModelInputs,
    wrt: LayerRef,
    output: OutputSelector | None,
    model_kwargs: Mapping[str, Any] | None,
) -> tuple[Tensor, Tensor]:
    _check_gradient_mode()
    if isinstance(wrt, str) and wrt == "embedding":
        get_embedding = getattr(model, "get_input_embeddings", None)
        if callable(get_embedding):
            wrt = get_embedding()
            if not isinstance(wrt, nn.Module):
                raise ValueError("get_input_embeddings() must return the model's embedding module.")
    name, layer = resolve_layer(model, wrt)
    activation = None

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], value: Any) -> Tensor:
        nonlocal activation
        if activation is not None:
            raise RuntimeError(f"Layer {name!r} ran more than once. Select a layer called once per forward.")
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(f"Layer {name!r} must return a floating-point tensor.")
        with torch.enable_grad():
            # A separate edge isolates this layer from parallel uses of the same tensor.
            activation = value.clone().requires_grad_()
            # Downstream in-place operations can modify this copy without changing the differentiation point.
            return activation.clone()

    with torch.enable_grad():
        with layer.register_forward_hook(capture):
            result = call_model(model, inputs, model_kwargs)
        value = select_tensor(result, output)
    if activation is None:
        raise RuntimeError(f"Layer {name!r} did not run during the model forward.")
    if not value.is_floating_point():
        raise TypeError("The selected output must be a floating-point tensor for layer derivatives.")
    return value, activation


def _layer_vjp(
    value: Tensor,
    activation: Tensor,
    cotangent: Tensor,
    *,
    batched: bool = False,
    retain_graph: bool = False,
) -> Tensor:
    if value.requires_grad:
        (projected,) = torch.autograd.grad(
            value, activation, cotangent, is_grads_batched=batched, retain_graph=retain_graph, allow_unused=True
        )
        if projected is not None:
            return projected
    shape = (cotangent.size(0), *activation.shape) if batched else activation.shape
    return activation.new_zeros(shape)


def _layer_jacobian(value: Tensor, activation: Tensor, chunk_size: int | None) -> Tensor:
    shape = (*value.shape, *activation.shape)
    size = value.numel()
    if size == 0 or not value.requires_grad:
        return activation.new_zeros(shape)

    rows = activation.new_empty((size, *activation.shape))
    chunk_size = size if chunk_size is None else chunk_size
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        indices = torch.arange(stop - start, device=value.device)
        basis = value.new_zeros((len(indices), size))
        basis[indices, indices + start] = 1
        retain_graph = stop < size
        if stop - start == 1:
            rows[start] = _layer_vjp(value, activation, basis.reshape(value.shape), retain_graph=retain_graph)
        else:
            rows[start:stop] = _layer_vjp(
                value, activation, basis.reshape(stop - start, *value.shape), batched=True, retain_graph=retain_graph
            )
    return rows.reshape(shape)
