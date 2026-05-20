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

from contextlib import suppress
from typing import Any, Mapping

import torch
from torch import Tensor
from transformers.pipelines.base import GenericTensor, Pipeline, PipelineException


class PolyadenylationPipeline(Pipeline):
    """
    Polyadenylation pipeline.

    The pipeline accepts raw DNA sequences and returns polyadenylation scores, such as APA isoform proportion or
    base-resolution cleavage probabilities.
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
        sequence = _resolve_sequence(inputs)
        tokenizer_kwargs = _tokenizer_kwargs(tokenizer_kwargs)
        return self.tokenizer(sequence, return_tensors=return_tensors, **tokenizer_kwargs)

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        outputs["input_ids"] = model_inputs["input_ids"]
        outputs["attention_mask"] = model_inputs.get("attention_mask")
        return outputs

    def postprocess(self, model_outputs):
        input_ids = model_outputs["input_ids"]
        attention_mask = model_outputs.get("attention_mask")
        sequences = _decode_sequences(self.tokenizer, input_ids, attention_mask)
        scores = _processed_scores(model_outputs, model=self.model, task=self.task)
        channels = _output_channels(self.model, scores)

        results = [
            _polyadenylation_result(sequence, sample_scores, channels)
            for sequence, sample_scores in zip(sequences, _sample_tensors(scores, len(sequences)))
        ]
        if len(results) == 1:
            return results[0]
        return results

    def _sanitize_parameters(self, tokenizer_kwargs: dict[str, Any] | None = None):
        preprocess_params: dict[str, Any] = {}
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs
        return preprocess_params, {}, {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            raise NotImplementedError("Only PyTorch is supported for polyadenylation.")


def _resolve_sequence(inputs: str | Mapping[str, Any]) -> str:
    if isinstance(inputs, Mapping):
        sequence = inputs.get("sequence")
    else:
        sequence = inputs
    if not sequence:
        raise ValueError("polyadenylation requires a DNA sequence.")
    return str(sequence)


def _processed_scores(outputs: Any, *, model, task: str) -> Tensor:
    if hasattr(model, "postprocess"):
        scores = model.postprocess(outputs)
    else:
        scores = _get_logits(outputs, task=task)
    if not isinstance(scores, Tensor):
        scores = torch.as_tensor(scores)
    return scores.detach().float().cpu()


def _get_logits(outputs: Any, *, task: str) -> Tensor:
    logits = outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", None)
    if logits is None:
        raise PipelineException(task, "model", "Unable to find logits in model outputs.")
    if not isinstance(logits, Tensor):
        logits = torch.as_tensor(logits)
    return logits


def _output_channels(model, scores: Tensor) -> list[str]:
    output_channels = getattr(model, "output_channels", None)
    if output_channels is not None:
        return list(output_channels)
    size = int(scores.shape[-1]) if scores.ndim > 0 else 1
    return [f"polyadenylation_{index}" for index in range(size)]


def _sample_tensors(scores: Tensor, sample_count: int) -> list[Tensor]:
    if sample_count == 1 and scores.ndim <= 1:
        return [scores]
    return list(scores)


def _polyadenylation_result(sequence: str, scores: Tensor, channels: list[str]) -> dict[str, Any]:
    values = scores.reshape(-1).tolist()
    result: dict[str, Any] = {"sequence": sequence}
    if len(values) == 1:
        result["score"] = float(values[0])
        result["channel"] = channels[0] if channels else "polyadenylation"
        return result
    if _is_cleavage_distribution(channels):
        result["cleavage_distribution"] = [
            _cleavage_distribution_row(channel, index, float(values[index]))
            for index, channel in enumerate(channels[: len(values)])
        ]
        return result
    result["channels"] = channels[: len(values)]
    result["scores"] = {channel: float(values[index]) for index, channel in enumerate(channels[: len(values)])}
    return result


def _is_cleavage_distribution(channels: list[str]) -> bool:
    return bool(channels) and (
        channels[-1] == "no_cleavage" or all(channel.startswith("cleavage_") for channel in channels)
    )


def _cleavage_distribution_row(channel: str, index: int, probability: float) -> dict[str, Any]:
    if channel == "no_cleavage":
        return {"event": "no_cleavage", "probability": probability}
    return {"position": _cleavage_position(channel, index), "probability": probability}


def _cleavage_position(channel: str, fallback: int) -> int:
    if channel.startswith("cleavage_"):
        with suppress(ValueError):
            return int(channel.removeprefix("cleavage_"))
    return fallback


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


__all__ = ["PolyadenylationPipeline"]
