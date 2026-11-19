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

"""Model calls, output selection, and layer lookup for interpretation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

from torch import Tensor, nn

TensorOrTuple: TypeAlias = Tensor | tuple[Tensor, ...]
ModelInputs: TypeAlias = Tensor | tuple[Any, ...] | Mapping[str, Any]
OutputSelector = str | Callable[[Any], Any]
LayerRef: TypeAlias = str | nn.Module


def call_model(model: nn.Module, inputs: ModelInputs, model_kwargs: Mapping[str, Any] | None = None) -> Any:
    kwargs = model_kwargs if model_kwargs is not None else {}
    if isinstance(inputs, Mapping):
        return model(**inputs, **kwargs)
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    return model(*args, **kwargs)


def select_output(value: Any, output: OutputSelector) -> Any:
    if callable(output):
        return output(value)
    if isinstance(value, Mapping):
        return value.get(output)
    return getattr(value, output, None)


def select_tensor(value: Any, output: OutputSelector | None = None) -> Tensor:
    if output is not None or not isinstance(value, Tensor):
        value = select_output(value, "logits" if output is None else output)
    if not isinstance(value, Tensor):
        raise TypeError("Use output to select a tensor from the model's result.")
    return value


def resolve_layer(model: nn.Module, layer: LayerRef) -> tuple[str, nn.Module]:
    if isinstance(layer, str):
        return layer, model.get_submodule(layer)
    for name, module in model.named_modules():
        if module is layer:
            return name, module
    raise ValueError("The selected layer must be a submodule of the supplied model.")


def resolve_layers(model: nn.Module, layers: Iterable[LayerRef]) -> dict[str, nn.Module]:
    return dict(resolve_layer(model, layer) for layer in layers)
