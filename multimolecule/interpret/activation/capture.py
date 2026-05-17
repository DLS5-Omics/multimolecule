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


from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from danling.data.utils import to_device
from torch import Tensor, nn

from ..outputs import ActivationOutput
from ..selectors import LayerSelector, resolve_modules
from ..utils import filter_forward_kwargs, get_model_device


def capture_activations(
    model: nn.Module,
    input_ids: Tensor,
    layers: LayerSelector,
    *,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    detach: bool = True,
    clone: bool = True,
    **forward_kwargs: Any,
) -> ActivationOutput:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}")

    module_selection = resolve_modules(model, layers)
    device = get_model_device(model)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    if position_ids is not None:
        position_ids = position_ids.to(device=device)
    forward_kwargs = to_device(forward_kwargs, device)

    activations: dict[str, Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            activations[name] = _normalize_activation(output, detach=detach, clone=clone)

        return hook

    for name, module in module_selection.modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    kwargs = filter_forward_kwargs(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "return_dict": True,
            **forward_kwargs,
        },
    )
    model.eval()
    try:
        with torch.no_grad() if detach else nullcontext():
            model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return ActivationOutput(
        activations=activations,
        requested_layers=module_selection.requested_layers,
        resolved_layers=module_selection.resolved_layers,
    )


def _normalize_activation(output: Any, *, detach: bool, clone: bool) -> Tensor:
    if isinstance(output, Tensor):
        tensor = output
    elif hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, Tensor):
        tensor = output.last_hidden_state
    elif isinstance(output, (tuple, list)):
        tensor_candidates = []
        for item in output:
            if isinstance(item, Tensor):
                tensor_candidates.append(item)
            elif hasattr(item, "last_hidden_state") and isinstance(item.last_hidden_state, Tensor):
                tensor_candidates.append(item.last_hidden_state)
            elif item is None:
                continue
        if len(tensor_candidates) == 1:
            tensor = tensor_candidates[0]
        elif len(tensor_candidates) > 1:
            raise TypeError(
                "Module output contains multiple tensors. Select a tensor-returning submodule path instead "
                "of relying on ambiguous tuple outputs."
            )
        else:
            raise TypeError("Module output does not contain a tensor activation")
    else:
        raise TypeError(f"Unsupported activation output type: {type(output)!r}")

    if detach:
        tensor = tensor.detach()
    if clone:
        tensor = tensor.clone()
    return tensor


__all__ = ["capture_activations"]
