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

import re
from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn

LayerSelector = str | Sequence[str]
_SELECTOR_ALIASES = ("embeddings", "attention_outputs", "mlp_outputs", "residual_outputs")
_RESIDUAL_BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:layer|layers|block|blocks|h)\.\d+$")


@dataclass
class ModuleSelection:
    requested_layers: list[str]
    resolved_layers: list[str]
    modules: dict[str, nn.Module]


def resolve_modules(model: nn.Module, layers: LayerSelector) -> ModuleSelection:
    if isinstance(layers, str):
        requested_layers = [layers]
    else:
        requested_layers = list(layers)
    if not requested_layers:
        raise ValueError("layers must contain at least one selector")

    named_modules = dict(model.named_modules())
    modules: dict[str, nn.Module] = {}
    for selector in requested_layers:
        for path in _resolve_selector(selector, named_modules):
            modules.setdefault(path, named_modules[path])
    return ModuleSelection(
        requested_layers=requested_layers,
        resolved_layers=list(modules),
        modules=modules,
    )


def _resolve_selector(selector: str, named_modules: dict[str, nn.Module]) -> list[str]:
    if selector in named_modules:
        return [selector]
    if selector == "embeddings":
        return _resolve_alias(selector, named_modules, _select_embedding_modules(named_modules))
    if selector == "attention_outputs":
        return _resolve_alias(selector, named_modules, _select_attention_output_modules(named_modules))
    if selector == "mlp_outputs":
        return _resolve_alias(selector, named_modules, _select_mlp_output_modules(named_modules))
    if selector == "residual_outputs":
        return _resolve_alias(selector, named_modules, _select_residual_output_modules(named_modules))
    available = ", ".join(list(named_modules)[:20])
    aliases = ", ".join(_SELECTOR_ALIASES)
    raise ValueError(
        f"Unknown layer selector {selector!r}. Expected an explicit module path or one of: {aliases}. "
        f"Example available modules: {available}"
    )


def _resolve_alias(selector: str, named_modules: dict[str, nn.Module], resolved_layers: list[str]) -> list[str]:
    if resolved_layers:
        return resolved_layers
    available = ", ".join(list(named_modules)[:20])
    raise ValueError(
        f"Layer selector {selector!r} did not match any modules on this model. "
        f"Example available modules: {available}"
    )


def _select_embedding_modules(named_modules: dict[str, nn.Module]) -> list[str]:
    embeddings = [name for name in named_modules if name and name.endswith(".embeddings")]
    if embeddings:
        return embeddings
    return [name for name in named_modules if name and name.endswith(".word_embeddings")]


def _select_attention_output_modules(named_modules: dict[str, nn.Module]) -> list[str]:
    wrapper_outputs = [
        name
        for name in named_modules
        if name and (name.endswith(".attention.output") or name.endswith(".crossattention.output"))
    ]
    if wrapper_outputs:
        return wrapper_outputs
    return [
        name
        for name in named_modules
        if name
        and (
            name.endswith(".self_attn.out_proj")
            or name.endswith(".self_attn.o_proj")
            or name.endswith(".attn.out_proj")
            or name.endswith(".attn.o_proj")
            or name.endswith(".attention.out_proj")
            or name.endswith(".crossattention.out_proj")
        )
    ]


def _select_mlp_output_modules(named_modules: dict[str, nn.Module]) -> list[str]:
    return [
        name
        for name in named_modules
        if name
        and (
            name.endswith(".mlp")
            or name.endswith(".ffn")
            or name.endswith(".feed_forward")
            or (name.endswith(".output") and ".attention.output" not in name and ".crossattention.output" not in name)
        )
    ]


def _select_residual_output_modules(named_modules: dict[str, nn.Module]) -> list[str]:
    return [name for name in named_modules if name and _RESIDUAL_BLOCK_PATTERN.search(name)]


__all__ = ["LayerSelector", "ModuleSelection", "resolve_modules"]
