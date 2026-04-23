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

from typing import Any

from lazy_imports import try_import
from torch import Tensor

from .base import ModelAttributor
from .registry import ATTRIBUTORS
from .types import AttributionTarget, Baseline

with try_import():
    from captum.attr import Saliency


@ATTRIBUTORS.register("saliency")
class SaliencyAttributor(ModelAttributor):
    method = "saliency"

    def _sanitize_parameters(
        self,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        abs_ = forward_kwargs.pop("abs", False)
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            target=target,
            baseline=baseline,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        forward_params["abs"] = abs_
        return preprocess_params, forward_params, postprocess_params

    def attribute(
        self,
        *,
        input_ids: Tensor,
        target: AttributionTarget,
        baseline: Baseline,
        attention_mask: Tensor,
        position_ids: Tensor | None,
        forward_kwargs: dict[str, Any],
        abs: bool = False,
    ) -> tuple[Tensor, None]:
        del baseline
        inputs_embeds = self.embedding_layer(input_ids)
        attribution = Saliency(self._forward_from_inputs_embeds)
        attributions = attribution.attribute(
            inputs=inputs_embeds,
            additional_forward_args=(target, attention_mask, position_ids, forward_kwargs),
            abs=abs,
        )
        return attributions, None
