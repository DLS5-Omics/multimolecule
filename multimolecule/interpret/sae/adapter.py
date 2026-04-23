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

from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn

from ..activation import capture_activations
from ..outputs import SaeOutput


@runtime_checkable
class SupportsEncode(Protocol):
    def encode(self, activations: Tensor) -> Tensor: ...


def run_sae(
    sae: SupportsEncode | nn.Module,
    model: nn.Module,
    input_ids: Tensor,
    *,
    layer: str,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    feature_ids: Tensor | None = None,
    **forward_kwargs: Any,
) -> SaeOutput:
    activation_output = capture_activations(
        model,
        input_ids,
        layer,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **forward_kwargs,
    )
    if len(activation_output.resolved_layers) != 1:
        raise ValueError("run_sae requires a single activation source")
    activations = activation_output.activations[activation_output.resolved_layers[0]]

    if isinstance(sae, SupportsEncode):
        features = sae.encode(activations)
    elif isinstance(sae, nn.Module):
        features = sae(activations)
    else:
        raise TypeError("sae must be an nn.Module or an object exposing encode()")

    if not isinstance(features, Tensor):
        raise TypeError(f"SAE returned unsupported output type: {type(features)!r}")

    return SaeOutput(features=features, feature_ids=feature_ids)


__all__ = ["run_sae", "SupportsEncode"]
