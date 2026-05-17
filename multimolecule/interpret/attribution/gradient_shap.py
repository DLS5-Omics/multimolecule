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
    from captum.attr import GradientShap


@ATTRIBUTORS.register("gradient_shap")
class GradientShapAttributor(ModelAttributor):
    method = "gradient_shap"

    def _sanitize_parameters(
        self,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        n_samples = forward_kwargs.pop("n_samples", 5)
        stdevs = forward_kwargs.pop("stdevs", 0.0)
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            target=target,
            baseline=baseline,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        forward_params.update(
            {
                "n_samples": n_samples,
                "stdevs": stdevs,
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
        n_samples: int = 5,
        stdevs: float = 0.0,
    ) -> tuple[Tensor, Tensor | None]:
        inputs_embeds = self.embedding_layer(input_ids)
        embed_baseline = self._resolve_embedding_baseline(input_ids, inputs_embeds, baseline)
        attribution = GradientShap(self._forward_from_inputs_embeds)
        return attribution.attribute(
            inputs=inputs_embeds,
            baselines=embed_baseline,
            n_samples=n_samples,
            stdevs=stdevs,
            additional_forward_args=(target, attention_mask, position_ids, forward_kwargs),
            return_convergence_delta=True,
        )
