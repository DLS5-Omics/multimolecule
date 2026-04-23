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

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from .targets import ScalarTarget


def filter_forward_kwargs(model: nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(model.forward)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in signature.parameters and value is not None}


def get_model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def get_model_config(model: nn.Module):
    if hasattr(model, "config"):
        return model.config
    nested_model = getattr(model, "model", None)
    if nested_model is not None and hasattr(nested_model, "config"):
        return nested_model.config
    raise ValueError("Unable to locate model config")


def get_input_embedding_layer(model: nn.Module, *, usage: str | None = None) -> nn.Module:
    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding
    nested_model = getattr(model, "model", None)
    if nested_model is not None and hasattr(nested_model, "get_input_embeddings"):
        embedding = nested_model.get_input_embeddings()
        if embedding is not None:
            return embedding
    if usage is None:
        raise ValueError("Model does not expose an input embedding layer")
    raise ValueError(f"Model does not expose an input embedding layer and cannot be used for {usage}")


def normalize_scalar_target(target: ScalarTarget | int | None) -> ScalarTarget:
    if target is None:
        return ScalarTarget()
    if isinstance(target, int):
        return ScalarTarget(class_idx=target)
    return target


def select_scalar_output(logits: Tensor, target: ScalarTarget, *, analysis_name: str) -> Tensor:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2:
        class_idx = _resolve_class_idx(logits, target)
        return logits[:, class_idx]
    if logits.ndim == 3:
        if target.token_index is None:
            raise ValueError(f"token_index must be specified for token-level {analysis_name} targets")
        class_idx = _resolve_class_idx(logits, target)
        return logits[:, target.token_index, class_idx]
    raise NotImplementedError(
        "Only sequence-level and token-level scalar targets are supported. "
        f"Contact or secondary-structure {analysis_name} is not implemented yet."
    )


def _resolve_class_idx(logits: Tensor, target: ScalarTarget) -> int:
    if logits.shape[-1] == 1:
        return 0
    if target.class_idx is None:
        raise ValueError("class_idx must be specified when logits have more than one output channel")
    return target.class_idx


__all__ = [
    "filter_forward_kwargs",
    "get_input_embedding_layer",
    "get_model_config",
    "get_model_device",
    "normalize_scalar_target",
    "select_scalar_output",
]
