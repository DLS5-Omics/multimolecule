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

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from danling.data.utils import to_device
from lazy_imports import try_import
from torch import Tensor, nn

from ..utils import (
    filter_forward_kwargs,
    get_input_embedding_layer,
    get_model_config,
    get_model_device,
    normalize_scalar_target,
    select_scalar_output,
)
from .types import AttributionMethod, AttributionOutput, AttributionTarget, Baseline

with try_import() as _captum_import:
    import captum.attr


class ModelAttributor(ABC):
    method: ClassVar[AttributionMethod]

    def __init__(self, model: nn.Module):
        _captum_import.check()
        self.model = model
        self.embedding_layer = get_input_embedding_layer(model, usage="Captum attribution")
        self.config = get_model_config(model)

    def __call__(
        self,
        input_ids: Tensor,
        target: AttributionTarget | int | None = None,
        *,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> AttributionOutput:
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            target=target,
            baseline=baseline,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        model_inputs = self.preprocess(input_ids, **preprocess_params)

        was_training = self.model.training
        self.model.eval()
        try:
            attributions, delta = self.attribute(**model_inputs, **forward_params)
        finally:
            self.model.train(was_training)

        return self.postprocess(attributions, delta=delta, **model_inputs, **postprocess_params)

    def _sanitize_parameters(
        self,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        preprocess_params = {
            "target": target,
            "baseline": baseline,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "forward_kwargs": forward_kwargs,
        }
        return preprocess_params, {}, {}

    def preprocess(
        self,
        input_ids: Tensor,
        target: AttributionTarget | int | None = None,
        baseline: Baseline = "pad",
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        forward_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}")

        input_ids = input_ids.to(device=self._device, dtype=torch.long)
        target = normalize_scalar_target(target)
        if attention_mask is None:
            attention_mask = self._default_attention_mask(input_ids)
        else:
            attention_mask = attention_mask.to(device=self._device)
        if position_ids is not None:
            position_ids = position_ids.to(device=self._device)
        if forward_kwargs is None:
            forward_kwargs = {}
        forward_kwargs = to_device(forward_kwargs, self._device)

        return {
            "input_ids": input_ids,
            "target": target,
            "baseline": baseline,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "forward_kwargs": forward_kwargs,
        }

    def postprocess(
        self,
        attributions: Tensor,
        *,
        delta: Tensor | None,
        target: AttributionTarget,
        baseline: Baseline,
        attention_mask: Tensor,
        **_: Any,
    ) -> AttributionOutput:
        attributions = attributions * attention_mask.unsqueeze(-1).to(attributions.dtype)
        token_attributions = attributions.sum(dim=-1)
        token_attributions = token_attributions * attention_mask.to(token_attributions.dtype)

        baseline_name = baseline if isinstance(baseline, str) else "custom"
        return AttributionOutput(
            attributions=attributions,
            token_attributions=token_attributions,
            method=self.method,
            target=target,
            baseline=baseline_name,
            delta=delta,
        )

    @abstractmethod
    def attribute(
        self,
        *,
        input_ids: Tensor,
        target: AttributionTarget,
        baseline: Baseline,
        attention_mask: Tensor,
        position_ids: Tensor | None,
        forward_kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor | None]: ...

    @property
    def _device(self) -> torch.device:
        return get_model_device(self.model)

    def _forward_from_input_ids(
        self,
        input_ids: Tensor,
        target: AttributionTarget,
        attention_mask: Tensor,
        position_ids: Tensor | None,
        forward_kwargs: dict[str, Any],
    ) -> Tensor:
        outputs = self._call_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            forward_kwargs=forward_kwargs,
        )
        return select_scalar_output(outputs.logits, target, analysis_name="attribution")

    def _forward_from_inputs_embeds(
        self,
        inputs_embeds: Tensor,
        target: AttributionTarget,
        attention_mask: Tensor,
        position_ids: Tensor | None,
        forward_kwargs: dict[str, Any],
    ) -> Tensor:
        outputs = self._call_model(
            inputs_embeds=inputs_embeds.clone(),
            attention_mask=attention_mask,
            position_ids=position_ids,
            forward_kwargs=forward_kwargs,
        )
        return select_scalar_output(outputs.logits, target, analysis_name="attribution")

    def _call_model(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        forward_kwargs: dict[str, Any],
    ):
        kwargs = dict(forward_kwargs)
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        kwargs["return_dict"] = True
        kwargs = filter_forward_kwargs(self.model, kwargs)
        return self.model(**kwargs)

    def _resolve_id_baseline(self, input_ids: Tensor, baseline: Baseline) -> Tensor | None:
        if isinstance(baseline, Tensor):
            if baseline.shape == input_ids.shape and not baseline.is_floating_point():
                return baseline.to(device=input_ids.device, dtype=torch.long)
            return None
        if baseline == "zero":
            return None
        if baseline == "pad":
            token_id = getattr(self.config, "pad_token_id", None)
        elif baseline == "mask":
            token_id = getattr(self.config, "mask_token_id", None)
        else:
            raise ValueError(f"Unsupported baseline: {baseline}")
        if token_id is None:
            raise ValueError(f"Model config does not define the token id required by baseline {baseline!r}")
        return torch.full_like(input_ids, fill_value=token_id)

    def _resolve_embedding_baseline(self, input_ids: Tensor, inputs_embeds: Tensor, baseline: Baseline) -> Tensor:
        if isinstance(baseline, Tensor):
            if baseline.shape == input_ids.shape and not baseline.is_floating_point():
                baseline = baseline.to(device=input_ids.device, dtype=torch.long)
                return self.embedding_layer(baseline)
            if baseline.shape == inputs_embeds.shape:
                return baseline.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            raise ValueError(
                "Custom baseline tensor must match either input_ids shape (batch, length) "
                "or embedding shape (batch, length, hidden_size)"
            )
        if baseline == "zero":
            return torch.zeros_like(inputs_embeds)
        baseline_ids = self._resolve_id_baseline(input_ids, baseline)
        if baseline_ids is None:
            raise ValueError(f"Unable to derive embedding baseline from baseline {baseline!r}")
        return self.embedding_layer(baseline_ids)

    def _resolve_feature_mask(self, feature_mask: Tensor | None, inputs_embeds: Tensor) -> Tensor:
        batch_size, length, hidden_size = inputs_embeds.shape
        if feature_mask is None:
            feature_mask = torch.arange(length, device=inputs_embeds.device).view(1, length, 1)
            return feature_mask.expand(1, length, hidden_size)

        feature_mask = feature_mask.to(device=inputs_embeds.device, dtype=torch.long)
        if feature_mask.shape == inputs_embeds.shape:
            return feature_mask
        if feature_mask.ndim == 2 and feature_mask.shape in {(1, length), (batch_size, length)}:
            return feature_mask.unsqueeze(-1).expand(*feature_mask.shape, hidden_size)
        if feature_mask.ndim == 3 and feature_mask.shape in {
            (1, length, hidden_size),
            (batch_size, length, hidden_size),
        }:
            return feature_mask
        raise ValueError(
            "feature_mask must match embedding shape (batch, length, hidden_size) " "or token shape (batch, length)"
        )

    def _default_attention_mask(self, input_ids: Tensor) -> Tensor:
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            return torch.ones_like(input_ids)
        return input_ids.ne(pad_token_id).to(dtype=torch.long)
