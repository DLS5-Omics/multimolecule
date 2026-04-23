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
    from captum.attr import FeatureAblation


@ATTRIBUTORS.register("feature_ablation")
class FeatureAblationAttributor(ModelAttributor):
    method = "feature_ablation"

    def _sanitize_parameters(
        self,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        feature_mask = forward_kwargs.pop("feature_mask", None)
        perturbations_per_eval = forward_kwargs.pop("perturbations_per_eval", 1)
        show_progress = forward_kwargs.pop("show_progress", False)
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            target=target,
            baseline=baseline,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        forward_params.update(
            {
                "feature_mask": feature_mask,
                "perturbations_per_eval": perturbations_per_eval,
                "show_progress": show_progress,
            }
        )
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
        feature_mask: Tensor | None = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> tuple[Tensor, None]:
        inputs_embeds = self.embedding_layer(input_ids)
        embed_baseline = self._resolve_embedding_baseline(input_ids, inputs_embeds, baseline)
        feature_mask = self._resolve_feature_mask(feature_mask, inputs_embeds)
        attribution = FeatureAblation(self._forward_from_inputs_embeds)
        attributions = attribution.attribute(
            inputs=inputs_embeds,
            baselines=embed_baseline,
            additional_forward_args=(target, attention_mask, position_ids, forward_kwargs),
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
        return attributions, None
