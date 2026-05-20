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

from inspect import signature
from typing import Any, Mapping

import torch
from torch import Tensor
from transformers.pipelines.base import GenericTensor, Pipeline


class MeanRibosomeLoadPipeline(Pipeline):
    """
    Mean-ribosome-load pipeline for 5'UTR models.

    The pipeline accepts raw nucleotide sequences and returns a sequence-level mean ribosome load (MRL) score.
    """

    def preprocess(
        self,
        inputs: str | Mapping[str, Any],
        return_tensors: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **preprocess_parameters,
    ) -> dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = "pt"
        sequence, extra_inputs = _resolve_inputs(inputs)
        tokenizer_kwargs = _tokenizer_kwargs(tokenizer_kwargs)
        model_inputs = self.tokenizer(sequence, return_tensors=return_tensors, **tokenizer_kwargs)
        model_inputs.update(extra_inputs)
        return model_inputs

    def _forward(self, model_inputs):
        outputs = _call_model(self.model, model_inputs)
        outputs["input_ids"] = model_inputs["input_ids"]
        outputs["attention_mask"] = model_inputs.get("attention_mask")
        return outputs

    def postprocess(self, model_outputs):
        input_ids = model_outputs["input_ids"]
        attention_mask = model_outputs.get("attention_mask")
        sequences = _decode_sequences(self.tokenizer, input_ids, attention_mask)
        scores = _get_scores(model_outputs)
        channels = _output_channels(self.model, scores)

        results = [
            _mean_ribosome_load_result(sequence, sample_scores, channels)
            for sequence, sample_scores in zip(sequences, _sample_tensors(scores, len(sequences)))
        ]
        if len(results) == 1:
            return results[0]
        return results

    def _sanitize_parameters(
        self,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        preprocess_params: dict[str, Any] = {}
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs
        return preprocess_params, {}, {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            raise NotImplementedError("Only PyTorch is supported for mean-ribosome-load.")


def _resolve_inputs(inputs: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(inputs, Mapping):
        sequence = inputs.get("sequence")
        extra_inputs = {key: value for key, value in inputs.items() if key != "sequence"}
    else:
        sequence = inputs
        extra_inputs = {}
    if not sequence:
        raise ValueError("mean-ribosome-load requires a nucleotide sequence.")
    return str(sequence), {key: _prepare_tensor(value) for key, value in extra_inputs.items()}


def _call_model(model, model_inputs: Mapping[str, Any]):
    return model(
        **{
            key: value
            for key, value in model_inputs.items()
            if key in {"input_ids", "attention_mask"} or _model_accepts_argument(model, key)
        }
    )


def _model_accepts_argument(model, argument: str) -> bool:
    try:
        parameters = signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return argument in parameters


def _prepare_tensor(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _get_scores(model_outputs: Any) -> Tensor:
    logits = model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs.get("logits")
    if logits is None:
        logits = model_outputs.logits_mfe if hasattr(model_outputs, "logits_mfe") else model_outputs.get("logits_mfe")
    if logits is None:
        raise ValueError("mean-ribosome-load requires model outputs with logits.")
    return logits.detach().float().cpu()


def _output_channels(model, scores: Tensor) -> list[str]:
    output_channels = getattr(model, "output_channels", None)
    if output_channels is not None:
        return list(output_channels)
    size = int(scores.shape[-1]) if scores.ndim > 0 else 1
    return [f"mean_ribosome_load_{index}" for index in range(size)]


def _sample_tensors(scores: Tensor, batch_size: int) -> list[Tensor]:
    if scores.ndim == 0:
        scores = scores.reshape(1, 1)
    elif scores.ndim == 1:
        scores = scores.unsqueeze(-1) if batch_size == scores.shape[0] else scores.unsqueeze(0)
    return [scores[index] for index in range(batch_size)]


def _mean_ribosome_load_result(sequence: str, scores: Tensor, channels: list[str]) -> dict[str, Any]:
    values = scores.reshape(-1).tolist()
    result: dict[str, Any] = {"sequence": sequence}
    if len(values) == 1:
        result["score"] = float(values[0])
        result["channel"] = channels[0] if channels else "mean_ribosome_load"
    else:
        result["channels"] = channels[: len(values)]
        result["scores"] = {channel: float(values[index]) for index, channel in enumerate(channels[: len(values)])}
    return result


def _decode_sequences(tokenizer, input_ids: Tensor, attention_mask: Tensor | None) -> list[str]:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    sequences = []
    for index, ids in enumerate(input_ids):
        if attention_mask is not None:
            ids = ids[attention_mask[index].bool()]
        sequences.append(tokenizer.decode(ids, skip_special_tokens=True).replace(" ", ""))
    return sequences


def _tokenizer_kwargs(tokenizer_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = {"add_special_tokens": False}
    if tokenizer_kwargs is not None:
        kwargs.update(tokenizer_kwargs)
    return kwargs


__all__ = ["MeanRibosomeLoadPipeline"]
