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

"""Adapt model calls and state for Captum attribution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass
from threading import RLock
from typing import Any, TypeAlias

import torch
from lazy_imports import LazyModule
from torch import Tensor, nn

from ..utils import LayerRef, OutputSelector, TensorOrTuple, resolve_layer, select_tensor
from .output import AttributionResult

AttributionInputs: TypeAlias = TensorOrTuple | Mapping[str, Any]
InputKeys: TypeAlias = str | Sequence[str]
Baselines: TypeAlias = Tensor | int | float | tuple[Tensor | int | float, ...]
Target: TypeAlias = int | tuple[int, ...] | list[int] | list[tuple[int, ...]] | Tensor

_CAPTUM_IMPORTS = LazyModule("from captum import attr", name=__name__, auto_all=False)
_CAPTUM_IMPORT_LOCK = RLock()


def _captum() -> Any:
    # LazyModule rejects a second request while the first resolution is pending.
    with _CAPTUM_IMPORT_LOCK:
        try:
            return _CAPTUM_IMPORTS.attr
        except ModuleNotFoundError as error:
            if error.name != "captum":
                raise
            raise ImportError("Attribution requires Captum. Install 'multimolecule[interpret]'.") from error


@dataclass(frozen=True)
class _StaticTensor:
    value: Tensor


def _unwrap_argument(value: Any) -> Any:
    return value.value if isinstance(value, _StaticTensor) else value


def _prepare_inputs(
    model: nn.Module,
    inputs: AttributionInputs,
    additional_forward_args: object,
    input_keys: InputKeys | None,
) -> tuple[Callable[..., Any], TensorOrTuple, object]:
    if not isinstance(inputs, Mapping):
        if input_keys is not None:
            raise ValueError("input_keys requires a mapping of model inputs.")
        return model, inputs, additional_forward_args

    if input_keys is None:
        raise ValueError("Specify input_keys when passing a mapping of model inputs.")
    if additional_forward_args is not None:
        raise ValueError("Include additional model arguments in the input mapping.")

    keys = (input_keys,) if isinstance(input_keys, str) else tuple(input_keys)
    if not keys:
        raise ValueError("Select at least one input key for attribution.")
    if len(set(keys)) != len(keys):
        raise ValueError("Attribution input keys must be unique.")

    values: list[Tensor] = []
    for key in keys:
        if key not in inputs:
            raise ValueError(f"Attribution input key {key!r} is missing from the input mapping.")
        value = inputs[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Attribution input {key!r} must be a tensor.")
        values.append(value)

    batch_size = values[0].size(0) if values[0].ndim else 0
    if not batch_size or any(not value.ndim or value.size(0) != batch_size for value in values):
        raise ValueError("Attributed tensors must have the same nonempty batch dimension.")

    fixed_keys = tuple(key for key in inputs if key not in keys)
    argument_names = (*keys, *fixed_keys)
    fixed_values = []
    for key in fixed_keys:
        value = inputs[key]
        if isinstance(value, Tensor) and (not value.ndim or value.size(0) != batch_size):
            # Captum expands only its top-level tensor arguments along the batch axis.
            value = _StaticTensor(value)
        fixed_values.append(value)

    def forward(*args: Any) -> Any:
        return model(**{key: _unwrap_argument(value) for key, value in zip(argument_names, args, strict=True)})

    tensor_inputs = values[0] if len(values) == 1 else tuple(values)
    return forward, tensor_inputs, tuple(fixed_values) or None


def _forward(model: Callable[..., Any], output: OutputSelector | None) -> Callable[..., Tensor]:
    def forward(*args: Any) -> Tensor:
        return select_tensor(model(*args), output)

    return forward


class _ModelForward(nn.Module):
    def __init__(self, model: nn.Module, forward: Callable[..., Any]) -> None:
        super().__init__()
        self.model = model
        self.training = model.training
        if isinstance(model, nn.DataParallel):
            self.device_ids = model.device_ids
        self._evaluate = forward

    def forward(self, *args: Any) -> Any:
        return self._evaluate(*args)


class _LRPModelForward(_ModelForward):
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        # Captum uses this snapshot to restore parameters changed by propagation rules.
        for name, value in state.items():
            if isinstance(value, Tensor):
                state[name] = value.detach().clone()
        return state


def _module_forward(
    model: nn.Module,
    forward: Callable[..., Any],
    output: OutputSelector | None = None,
    *,
    raw_output: bool = False,
    paired_inputs: bool = False,
) -> nn.Module:
    if paired_inputs and isinstance(model, nn.DataParallel) and len(model.device_ids) > 1:
        raise ValueError("DeepLIFT methods require input and baseline pairs on a single model replica.")

    def evaluate(*args: Any) -> Any:
        value = forward(*args)
        return value if raw_output else select_tensor(value, output)

    return _ModelForward(model, evaluate)


def _lrp_forward(
    model: nn.Module,
    forward: Callable[..., Any],
    additional_forward_args: object,
    output: OutputSelector | None = None,
) -> nn.Module:
    fixed = additional_forward_args if isinstance(additional_forward_args, tuple) else (additional_forward_args,)
    if additional_forward_args is None:
        fixed = ()

    def prepare(value: Any) -> Any:
        if isinstance(value, _StaticTensor):
            return _StaticTensor(prepare(value.value))
        if isinstance(value, Tensor):
            value = value.clone()
            if value.is_floating_point() and not value.requires_grad:
                value.requires_grad_(True)
        return value

    def evaluate(*inputs: Any) -> Any:
        # LRP adds zero to the inputs used by its second forward. Bind fixed model arguments separately.
        return forward(*(prepare(value) for value in (*inputs, *fixed)))

    return _LRPModelForward(model, _forward(evaluate, output))


@contextmanager
def _isolated_relu_inputs(model: nn.Module) -> Iterator[None]:
    def isolate(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(value.clone() if isinstance(value, Tensor) else value for value in inputs)

    # Captum's ReLU hooks attach backward hooks to the input tensor itself.
    with ExitStack() as hooks:
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.enter_context(module.register_forward_pre_hook(isolate))
        yield


@contextmanager
def _preserve_module_attributes(model: nn.Module, *names: str) -> Iterator[None]:
    states = []
    for module in model.modules():
        attributes = vars(module)
        original = {name: attributes[name] for name in names if name in attributes}
        parameters = {
            name: value
            for name, value in module.named_parameters(recurse=False, remove_duplicate=False)
            if name in names
        }
        buffers = {
            name: value for name, value in module.named_buffers(recurse=False, remove_duplicate=False) if name in names
        }
        persistent = set(buffers).intersection(module.state_dict()) if buffers else set()
        states.append((module, original, parameters, buffers, persistent))

    try:
        yield
    finally:
        for module, original, parameters, buffers, persistent in states:
            attributes = vars(module)
            for name in names:
                if name in original:
                    attributes[name] = original[name]
                else:
                    attributes.pop(name, None)
            for name, parameter in parameters.items():
                if getattr(module, name, None) is not parameter:
                    if hasattr(module, name):
                        delattr(module, name)
                    module.register_parameter(name, parameter)
            for name, buffer in buffers.items():
                if getattr(module, name, None) is not buffer:
                    if hasattr(module, name):
                        delattr(module, name)
                    module.register_buffer(name, buffer, persistent=name in persistent)


@contextmanager
def _lrp_state(model: nn.Module) -> Iterator[None]:
    with _preserve_module_attributes(model, "rule", "activations", "activation"):
        for module in model.modules():
            rule = getattr(module, "rule", None)
            if rule is not None:
                # Captum stores activation and relevance caches on the rule during attribution.
                module.rule = copy(rule)
        yield


def _layer_modules(model: nn.Module, layer: LayerRef | list[LayerRef]) -> nn.Module | list[nn.Module]:
    if not isinstance(layer, list):
        return resolve_layer(model, layer)[1]
    if not layer:
        raise ValueError("Select at least one layer.")
    modules = [resolve_layer(model, item)[1] for item in layer]
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("Each layer must refer to a different module.")
    return modules


def _device_ids(model: nn.Module) -> list[int] | None:
    return (model.device_ids or None) if isinstance(model, nn.DataParallel) else None


def _gradient_result(
    attribute: Callable[..., Any],
    inputs: TensorOrTuple,
    return_convergence_delta: bool = False,
    **kwargs: Any,
) -> AttributionResult:
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Run gradient attribution outside torch.inference_mode().")
    input_tensors = (inputs,) if isinstance(inputs, Tensor) else inputs
    forward_args = kwargs.get("additional_forward_args")
    arguments = forward_args if isinstance(forward_args, tuple) else (forward_args,)
    supplied = (*input_tensors, *map(_unwrap_argument, arguments))
    grad_flags = [(value, value.requires_grad) for value in supplied if isinstance(value, Tensor)]
    try:
        if return_convergence_delta:
            attributions, delta = attribute(inputs, return_convergence_delta=True, **kwargs)
            return AttributionResult(attributions, delta)
        return AttributionResult(attribute(inputs, **kwargs))
    finally:
        for tensor, requires_grad in grad_flags:
            if tensor.requires_grad != requires_grad:
                tensor.requires_grad_(requires_grad)
