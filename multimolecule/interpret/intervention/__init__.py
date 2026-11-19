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

"""Compare model outputs after replacing named submodule activations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..utils import LayerRef, ModelInputs, OutputSelector, call_model, resolve_layer, select_tensor

__all__ = ["InterventionResult", "compare"]


@dataclass(frozen=True)
class InterventionResult:
    """The selected model outputs before and after an intervention."""

    original: Tensor
    intervened: Tensor

    @property
    def delta(self) -> Tensor:
        """The intervened output minus the original output."""
        return self.intervened - self.original


@torch.no_grad()
def compare(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    replacements: Mapping[LayerRef, Callable[[Any], Any]],
    output: OutputSelector | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> InterventionResult:
    """Run an original forward and a second forward with replacement activations.

    Each key in ``replacements`` is a submodule object or qualified name. Its transform
    receives that submodule's output on every invocation and returns the replacement. For example,
    pass ``torch.zeros_like`` to ablate a tensor-valued activation. To patch an activation,
    return a tensor captured from a separate run.

    Both forwards use the same inputs. By default ``output`` selects the returned tensor or
    the model's ``logits``. A field name or callable can select another tensor. Use an
    evaluation-mode model for deterministic comparisons and use the model exclusively
    during the intervention while the hooks are installed.
    """
    layers = [(*resolve_layer(model, layer), transform) for layer, transform in replacements.items()]
    if len({id(module) for _, module, _ in layers}) != len(layers):
        raise ValueError("Multiple replacements refer to the same submodule.")
    original = select_tensor(call_model(model, inputs, model_kwargs), output).clone()
    applied: set[str] = set()

    def hook_for(name: str, transform: Callable[[Any], Any]) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def replace(_module: nn.Module, _args: tuple[Any, ...], value: Any) -> Any:
            replacement = transform(value)
            if replacement is None:
                raise TypeError(f"Intervention for {name!r} must return a replacement.")
            applied.add(name)
            return replacement

        return replace

    with ExitStack() as stack:
        for name, module, transform in layers:
            handle = module.register_forward_hook(hook_for(name, transform))
            stack.callback(handle.remove)
        intervened = select_tensor(call_model(model, inputs, model_kwargs), output).clone()

    missing = {name for name, _, _ in layers} - applied
    if missing:
        raise RuntimeError(f"Intervention layers did not run: {', '.join(sorted(missing))}.")
    if original.shape != intervened.shape:
        raise ValueError("Original and intervened outputs must have the same shape.")
    return InterventionResult(original, intervened)
