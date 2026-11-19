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

"""Capture the intermediate activations of PyTorch models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from functools import partial
from typing import Any

from torch import Tensor, nn

from ..utils import LayerRef, ModelInputs, call_model, resolve_layers

__all__ = ["capture"]


def _snapshot(value: Any, detach: bool) -> Any:
    if isinstance(value, Tensor):
        return (value.detach() if detach else value).clone()
    if isinstance(value, tuple):
        return tuple(_snapshot(item, detach) for item in value)
    if isinstance(value, list):
        return [_snapshot(item, detach) for item in value]
    if isinstance(value, Mapping):
        return {key: _snapshot(item, detach) for key, item in value.items()}
    return value


def capture(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    layers: LayerRef | Sequence[LayerRef],
    model_kwargs: Mapping[str, Any] | None = None,
    detach: bool = True,
) -> dict[str, tuple[Any, ...]]:
    """Capture outputs from selected submodules during one model call.

    Pass dotted submodule paths or module objects. Each result key is a dotted
    submodule path, with one snapshot per invocation in forward order.
    A skipped submodule has an empty tuple.
    Tensor outputs, including those in tuples, lists and mappings, are cloned
    when captured. Wrap evaluation calls in ``torch.no_grad()`` to save graph
    memory, or pass ``detach=False`` to keep the snapshots connected to the graph.

    Hooks are active for this model call. Use the model exclusively during
    capture to keep the outputs from a single forward together.
    """
    modules = resolve_layers(model, (layers,) if isinstance(layers, (str, nn.Module)) else layers)
    outputs: dict[str, list[Any]] = {name: [] for name in modules}

    def collect(name: str, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        outputs[name].append(_snapshot(output, detach))

    with ExitStack() as hooks:
        for name, module in modules.items():
            hooks.enter_context(module.register_forward_hook(partial(collect, name)))
        call_model(model, inputs, model_kwargs)

    return {name: tuple(values) for name, values in outputs.items()}
