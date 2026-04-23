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
from torch import Tensor, nn

from .base import ModelAttributor
from .registry import ATTRIBUTORS
from .types import AttributionTarget, Baseline

with try_import():
    from captum.attr import Occlusion


@ATTRIBUTORS.register("occlusion")
class OcclusionAttributor(ModelAttributor):
    method = "occlusion"

    def __init__(
        self,
        model: nn.Module,
        *,
        sliding_window_shapes: tuple[int, int] | None = None,
        strides: tuple[int, int] | None = None,
    ):
        super().__init__(model)
        self.sliding_window_shapes = sliding_window_shapes
        self.strides = strides

    def _sanitize_parameters(
        self,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        sliding_window_shapes = forward_kwargs.pop("sliding_window_shapes", self.sliding_window_shapes)
        strides = forward_kwargs.pop("strides", self.strides)
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            target=target,
            baseline=baseline,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        forward_params.update({"sliding_window_shapes": sliding_window_shapes, "strides": strides})
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
        sliding_window_shapes: tuple[int, int] | None,
        strides: tuple[int, int] | None,
    ) -> tuple[Tensor, None]:
        inputs_embeds = self.embedding_layer(input_ids)
        embed_baseline = self._resolve_embedding_baseline(input_ids, inputs_embeds, baseline)
        hidden_size = inputs_embeds.shape[-1]
        attribution = Occlusion(self._forward_from_inputs_embeds)
        attributions = attribution.attribute(
            inputs=inputs_embeds,
            baselines=embed_baseline,
            sliding_window_shapes=sliding_window_shapes or (1, hidden_size),
            strides=strides or (1, hidden_size),
            additional_forward_args=(target, attention_mask, position_ids, forward_kwargs),
        )
        return attributions, None
