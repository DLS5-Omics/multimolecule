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
    from captum.attr import IntegratedGradients, LayerIntegratedGradients


@ATTRIBUTORS.register("layer_integrated_gradients")
class LayerIntegratedGradientsAttributor(ModelAttributor):
    method = "layer_integrated_gradients"

    def __init__(self, model: nn.Module, *, n_steps: int = 50, internal_batch_size: int | None = None):
        super().__init__(model)
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def _sanitize_parameters(
        self,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        n_steps = forward_kwargs.pop("n_steps", self.n_steps)
        internal_batch_size = forward_kwargs.pop("internal_batch_size", self.internal_batch_size)
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            target=target,
            baseline=baseline,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        forward_params.update({"n_steps": n_steps, "internal_batch_size": internal_batch_size})
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
        n_steps: int,
        internal_batch_size: int | None,
    ) -> tuple[Tensor, Tensor | None]:
        id_baseline = self._resolve_id_baseline(input_ids, baseline)
        if id_baseline is not None:
            attribution = LayerIntegratedGradients(self._forward_from_input_ids, self.embedding_layer)
            return attribution.attribute(
                inputs=input_ids,
                baselines=id_baseline,
                additional_forward_args=(target, attention_mask, position_ids, forward_kwargs),
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )

        inputs_embeds = self.embedding_layer(input_ids)
        embed_baseline = self._resolve_embedding_baseline(input_ids, inputs_embeds, baseline)
        attribution = IntegratedGradients(self._forward_from_inputs_embeds)
        return attribution.attribute(
            inputs=inputs_embeds,
            baselines=embed_baseline,
            additional_forward_args=(target, attention_mask, position_ids, forward_kwargs),
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
