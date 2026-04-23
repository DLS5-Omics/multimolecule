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

from dataclasses import dataclass

from torch import Tensor

from .targets import ScalarTarget


@dataclass
class AttributionOutput:
    attributions: Tensor
    token_attributions: Tensor
    method: str
    target: ScalarTarget
    baseline: str
    delta: Tensor | None = None


@dataclass
class ActivationOutput:
    activations: dict[str, Tensor]
    requested_layers: list[str]
    resolved_layers: list[str]


@dataclass
class AttentionOutput:
    attentions: Tensor
    tokens: list[str] | None = None
    layers: list[int] | None = None
    heads: list[int] | None = None
    aggregation: str | None = None


@dataclass
class JacobianOutput:
    scores: Tensor
    target: ScalarTarget
    positions: Tensor | None = None
    reduction: str = "none"
    top_k_indices: Tensor | None = None
    top_k_scores: Tensor | None = None


@dataclass
class SaeOutput:
    features: Tensor
    feature_ids: Tensor | None = None


__all__ = [
    "ActivationOutput",
    "AttentionOutput",
    "AttributionOutput",
    "JacobianOutput",
    "SaeOutput",
]
