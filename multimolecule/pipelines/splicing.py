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

import math
from typing import Any, Mapping

import torch
from torch import Tensor
from transformers.pipelines.base import GenericTensor, Pipeline, PipelineException


class SpliceSitePredictionPipeline(Pipeline):
    """
    Splice-site prediction pipeline for DNA/RNA splicing models.

    The pipeline accepts raw nucleotide sequences and returns biological position-level scores. It supports classical
    fixed-window splice-site scorers such as MaxEntScan and per-nucleotide models such as OpenSpliceAI, Pangolin, and
    SpTransformer.
    """

    threshold: float = 0.5
    output_scores: bool = True
    top_k: int | None = None

    def preprocess(
        self,
        inputs: str,
        return_tensors: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **preprocess_parameters,
    ) -> dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = "pt"
        tokenizer_kwargs = _tokenizer_kwargs(tokenizer_kwargs)
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        outputs["input_ids"] = model_inputs["input_ids"]
        outputs["attention_mask"] = model_inputs.get("attention_mask")
        return outputs

    def postprocess(
        self,
        model_outputs,
        threshold: float | None = None,
        output_scores: bool | None = None,
        top_k: int | None = None,
    ):
        if threshold is None:
            threshold = self.threshold
        if output_scores is None:
            output_scores = self.output_scores
        if top_k is None:
            top_k = self.top_k

        logits = _get_logits(model_outputs, task="splice-site-prediction")
        input_ids = model_outputs["input_ids"]
        attention_mask = model_outputs.get("attention_mask")
        sequences = _decode_sequences(self.tokenizer, input_ids, attention_mask)
        config = getattr(self.model, "config", None)

        results = []
        for sequence, sample_logits in zip(sequences, logits):
            results.append(
                _splice_site_result(
                    sequence,
                    sample_logits,
                    model=self.model,
                    config=config,
                    threshold=threshold,
                    output_scores=output_scores,
                    top_k=top_k,
                )
            )
        if len(results) == 1:
            return results[0]
        return results

    def _sanitize_parameters(
        self,
        threshold: float | None = None,
        output_scores: bool | None = None,
        top_k: int | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        preprocess_params: dict[str, Any] = {}
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs

        postprocess_params: dict[str, Any] = {}
        if threshold is not None:
            _check_probability("threshold", threshold, self.task, _model_prefix(self.model))
            postprocess_params["threshold"] = threshold
        if output_scores is not None:
            if not isinstance(output_scores, bool):
                raise PipelineException(
                    self.task,
                    _model_prefix(self.model),
                    f"output_scores must be a boolean, got {type(output_scores)}.",
                )
            postprocess_params["output_scores"] = output_scores
        if top_k is not None:
            if top_k <= 0:
                raise PipelineException(self.task, _model_prefix(self.model), "top_k must be a positive integer.")
            postprocess_params["top_k"] = top_k
        return preprocess_params, {}, postprocess_params

    def __init__(self, *args, threshold: float | None = None, output_scores: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            raise NotImplementedError("Only PyTorch is supported for splice-site prediction.")
        if threshold is not None:
            _check_probability("threshold", threshold, self.task, _model_prefix(self.model))
            self.threshold = threshold
        if output_scores is not None:
            if not isinstance(output_scores, bool):
                raise PipelineException(
                    self.task,
                    _model_prefix(self.model),
                    f"output_scores must be a boolean, got {type(output_scores)}.",
                )
            self.output_scores = output_scores


class SpliceVariantEffectPipeline(Pipeline):
    """
    Splice variant-effect pipeline.

    The pipeline scores a reference sequence and an alternative sequence and returns the alternative-minus-reference
    delta. Models with a native paired-input variant-effect head, such as MMSplice and MTSplice, are called once with
    both sequences. Other models are scored on reference and alternative separately.
    """

    top_k: int | None = 20

    def preprocess(
        self,
        inputs: str | Mapping[str, str],
        alternative: str | None = None,
        return_tensors: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **preprocess_parameters,
    ) -> dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = "pt"
        reference, alternative = _resolve_variant_inputs(inputs, alternative)
        tokenizer_kwargs = _tokenizer_kwargs(tokenizer_kwargs)
        reference_inputs = self.tokenizer(reference, return_tensors=return_tensors, **tokenizer_kwargs)
        alternative_inputs = self.tokenizer(alternative, return_tensors=return_tensors, **tokenizer_kwargs)
        return {
            **reference_inputs,
            "reference_sequence": reference,
            "alternative_sequence": alternative,
            "alternative_input_ids": alternative_inputs["input_ids"],
            "alternative_attention_mask": alternative_inputs.get("attention_mask"),
        }

    def _forward(self, model_inputs):
        reference_sequence = model_inputs.pop("reference_sequence")
        alternative_sequence = model_inputs.pop("alternative_sequence")
        alternative_input_ids = model_inputs.pop("alternative_input_ids")
        alternative_attention_mask = model_inputs.pop("alternative_attention_mask", None)

        model_type = _model_type(self.model)
        if model_type in {"mmsplice", "mtsplice"}:
            outputs = self.model(
                **model_inputs,
                alternative_input_ids=alternative_input_ids,
                alternative_attention_mask=alternative_attention_mask,
            )
            outputs["reference_sequence"] = reference_sequence
            outputs["alternative_sequence"] = alternative_sequence
            return outputs

        reference_outputs = self.model(**model_inputs)
        alternative_inputs = {
            "input_ids": alternative_input_ids,
            "attention_mask": alternative_attention_mask,
        }
        alternative_outputs = self.model(
            **{key: value for key, value in alternative_inputs.items() if value is not None}
        )
        return {
            "reference_outputs": reference_outputs,
            "alternative_outputs": alternative_outputs,
            "reference_sequence": reference_sequence,
            "alternative_sequence": alternative_sequence,
        }

    def postprocess(self, model_outputs, top_k: int | None = None):
        if top_k is None:
            top_k = self.top_k
        reference_sequence = model_outputs["reference_sequence"]
        alternative_sequence = model_outputs["alternative_sequence"]
        config = getattr(self.model, "config", None)

        if "reference_outputs" in model_outputs:
            reference_scores, channels = _processed_scores(
                _get_logits(model_outputs["reference_outputs"], task="splice-variant-effect"),
                model=self.model,
                config=config,
            )
            alternative_scores, _ = _processed_scores(
                _get_logits(model_outputs["alternative_outputs"], task="splice-variant-effect"),
                model=self.model,
                config=config,
            )
            delta_scores = alternative_scores - reference_scores
            return _variant_effect_result(
                reference_sequence,
                alternative_sequence,
                delta_scores,
                channels,
                reference_scores=reference_scores,
                alternative_scores=alternative_scores,
                top_k=top_k,
            )

        delta_scores, channels = _processed_scores(
            _get_logits(model_outputs, task="splice-variant-effect"),
            model=self.model,
            config=config,
        )
        return _variant_effect_result(reference_sequence, alternative_sequence, delta_scores, channels, top_k=top_k)

    def _sanitize_parameters(
        self,
        alternative: str | None = None,
        top_k: int | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        preprocess_params: dict[str, Any] = {}
        if alternative is not None:
            preprocess_params["alternative"] = alternative
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs
        postprocess_params: dict[str, Any] = {}
        if top_k is not None:
            if top_k <= 0:
                raise PipelineException(self.task, _model_prefix(self.model), "top_k must be a positive integer.")
            postprocess_params["top_k"] = top_k
        return preprocess_params, {}, postprocess_params

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            raise NotImplementedError("Only PyTorch is supported for splice variant-effect prediction.")

    def __call__(self, inputs, alternative: str | None = None, **kwargs):
        if alternative is not None:
            kwargs["alternative"] = alternative
        return super().__call__(inputs, **kwargs)


def _splice_site_result(
    sequence: str,
    logits: Tensor,
    *,
    model: Any,
    config: Any,
    threshold: float,
    output_scores: bool,
    top_k: int | None,
) -> dict[str, Any]:
    scores, channels = _processed_scores(logits, model=model, config=config)
    if scores.ndim == 1:
        return {
            "sequence": sequence,
            "score": _score_value(scores),
            "position_index_base": 0,
            "channels": channels,
        }

    scores = scores[: len(sequence)]
    result: dict[str, Any] = {
        "sequence": sequence,
        "position_index_base": 0,
        "splice_sites": _splice_site_calls(sequence, scores, channels, threshold=threshold, top_k=top_k),
        "channels": channels,
    }
    if output_scores:
        result["scores"] = _position_scores(sequence, scores, channels)
    return result


def _variant_effect_result(
    reference_sequence: str,
    alternative_sequence: str,
    delta_scores: Tensor,
    channels: list[str],
    *,
    reference_scores: Tensor | None = None,
    alternative_scores: Tensor | None = None,
    top_k: int | None = 20,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_sequence": reference_sequence,
        "alternative_sequence": alternative_sequence,
        "position_index_base": 0,
        "channels": channels,
    }
    if delta_scores.ndim == 1:
        result["delta_score"] = _score_value(delta_scores)
        if reference_scores is not None:
            result["reference_score"] = _score_value(reference_scores)
        if alternative_scores is not None:
            result["alternative_score"] = _score_value(alternative_scores)
        result["variant_effects"] = _sequence_delta_effects(
            delta_scores,
            channels,
            reference_scores=reference_scores,
            alternative_scores=alternative_scores,
            top_k=top_k,
        )
        return result

    length = min(len(reference_sequence), delta_scores.shape[0])
    result["delta_scores"] = _position_scores(reference_sequence[:length], delta_scores[:length], channels)
    if reference_scores is not None:
        result["reference_scores"] = _position_scores(reference_sequence[:length], reference_scores[:length], channels)
    if alternative_scores is not None:
        result["alternative_scores"] = _position_scores(
            alternative_sequence[:length], alternative_scores[:length], channels
        )
    result["variant_effects"] = _position_delta_effects(
        reference_sequence[:length],
        alternative_sequence[:length],
        delta_scores[:length],
        channels,
        reference_scores=reference_scores[:length] if reference_scores is not None else None,
        alternative_scores=alternative_scores[:length] if alternative_scores is not None else None,
        top_k=top_k,
    )
    return result


def _processed_scores(logits: Tensor, *, model: Any, config: Any) -> tuple[Tensor, list[str]]:
    channels = None
    postprocess = getattr(model, "postprocess", None)
    if callable(postprocess):
        processed = postprocess(logits)
        if isinstance(processed, tuple) and len(processed) == 2:
            logits, channels = processed
        else:
            logits = processed

    logits = logits.detach().float().cpu()
    if logits.ndim > 1 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    if logits.ndim == 1:
        return logits, list(channels) if channels is not None else _sequence_channels(logits.shape[-1], config)

    num_channels = logits.shape[-1]
    if channels is not None:
        return logits, list(channels)
    if num_channels == 3:
        return logits.softmax(dim=-1), _splice_channels()
    if num_channels == 2:
        return logits.softmax(dim=-1), ["no_splice", "splice_site"]
    return logits, _sequence_channels(num_channels, config)


def _splice_site_calls(
    sequence: str,
    scores: Tensor,
    channels: list[str],
    *,
    threshold: float,
    top_k: int | None,
) -> list[dict[str, Any]]:
    labels = [
        label for label in channels if label in {"acceptor", "donor", "splice_site"} or label.endswith("_splice_site")
    ]
    calls: list[dict[str, Any]] = []
    for position in range(scores.shape[0]):
        for label in labels:
            score = float(scores[position, channels.index(label)].item())
            if score >= threshold:
                calls.append(
                    {
                        "position": position,
                        "nucleotide": sequence[position] if position < len(sequence) else None,
                        "type": label,
                        "score": score,
                    }
                )
    calls.sort(key=lambda item: float(item["score"]), reverse=True)
    if top_k is not None:
        return calls[:top_k]
    return calls


def _position_scores(sequence: str, scores: Tensor, channels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in range(scores.shape[0]):
        row: dict[str, Any] = {
            "position": position,
            "nucleotide": sequence[position] if position < len(sequence) else None,
        }
        row.update({channel: float(scores[position, index].item()) for index, channel in enumerate(channels)})
        rows.append(row)
    return rows


def _position_delta_effects(
    reference_sequence: str,
    alternative_sequence: str,
    delta_scores: Tensor,
    channels: list[str],
    *,
    reference_scores: Tensor | None,
    alternative_scores: Tensor | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    effects = []
    for position in range(delta_scores.shape[0]):
        for channel_index, channel in enumerate(channels[: delta_scores.shape[-1]]):
            if _is_background_channel(channel):
                continue
            delta_score = _finite_float(delta_scores[position, channel_index])
            if delta_score is None:
                continue
            effect: dict[str, Any] = {
                "position": position,
                "reference_nucleotide": reference_sequence[position] if position < len(reference_sequence) else None,
                "alternative_nucleotide": (
                    alternative_sequence[position] if position < len(alternative_sequence) else None
                ),
                "channel": channel,
                "delta_score": delta_score,
                "direction": _delta_direction(delta_score),
            }
            if reference_scores is not None:
                effect["reference_score"] = _finite_float(reference_scores[position, channel_index])
            if alternative_scores is not None:
                effect["alternative_score"] = _finite_float(alternative_scores[position, channel_index])
            effects.append(effect)
    return _rank_delta_effects(effects, top_k)


def _sequence_delta_effects(
    delta_scores: Tensor,
    channels: list[str],
    *,
    reference_scores: Tensor | None,
    alternative_scores: Tensor | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    effects = []
    for channel_index, channel in enumerate(channels[: delta_scores.shape[-1]]):
        if _is_background_channel(channel):
            continue
        delta_score = _finite_float(delta_scores[channel_index])
        if delta_score is None:
            continue
        effect: dict[str, Any] = {
            "position": None,
            "channel": channel,
            "delta_score": delta_score,
            "direction": _delta_direction(delta_score),
        }
        if reference_scores is not None:
            effect["reference_score"] = _finite_float(reference_scores[channel_index])
        if alternative_scores is not None:
            effect["alternative_score"] = _finite_float(alternative_scores[channel_index])
        effects.append(effect)
    return _rank_delta_effects(effects, top_k)


def _rank_delta_effects(effects: list[dict[str, Any]], top_k: int | None) -> list[dict[str, Any]]:
    effects.sort(key=lambda effect: abs(float(effect["delta_score"])), reverse=True)
    if top_k is not None:
        return effects[:top_k]
    return effects


def _finite_float(value: Tensor) -> float | None:
    score = float(value.item())
    if math.isfinite(score):
        return score
    return None


def _delta_direction(delta_score: float) -> str:
    if delta_score > 0:
        return "gain"
    if delta_score < 0:
        return "loss"
    return "unchanged"


def _is_background_channel(channel: str) -> bool:
    return channel == "no_splice" or channel.endswith("_no_splice")


def _get_logits(outputs: Any, *, task: str) -> Tensor:
    logits = outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", None)
    if logits is None:
        raise PipelineException(task, "model", "Unable to find logits in model outputs.")
    if not isinstance(logits, Tensor):
        logits = torch.as_tensor(logits)
    return logits


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


def _resolve_variant_inputs(inputs: str | Mapping[str, str], alternative: str | None) -> tuple[str, str]:
    if isinstance(inputs, Mapping):
        reference = inputs.get("reference") or inputs.get("reference_sequence") or inputs.get("sequence")
        alternative = alternative or inputs.get("alternative") or inputs.get("alternative_sequence")
    else:
        reference = inputs
    if not reference:
        raise ValueError("splice-variant-effect requires a reference sequence.")
    if not alternative:
        raise ValueError("splice-variant-effect requires an alternative sequence.")
    if len(reference) != len(alternative):
        raise ValueError(
            "splice-variant-effect currently requires reference and alternative sequences with the same length."
        )
    return reference, alternative


def _tokenizer_kwargs(tokenizer_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = {"add_special_tokens": False}
    if tokenizer_kwargs is not None:
        kwargs.update(tokenizer_kwargs)
    return kwargs


def _check_probability(name: str, value: float, task: str, model_prefix: str) -> None:
    if value <= 0 or value >= 1:
        raise PipelineException(task, model_prefix, f"{name} must be between 0 and 1, got {value}.")


def _model_type(model) -> str:
    return str(getattr(getattr(model, "config", None), "model_type", "")).lower()


def _model_prefix(model) -> str:
    return str(getattr(model, "base_model_prefix", "model"))


def _num_tissues(config: Any, default: int) -> int:
    return int(getattr(config, "num_tissues", default) or default)


def _splice_channels() -> list[str]:
    return ["no_splice", "acceptor", "donor"]


def _tissue_channels(num_tissues: int) -> list[str]:
    return [f"tissue_{index}" for index in range(num_tissues)]


def _sequence_channels(num_channels: int, config: Any) -> list[str]:
    if num_channels == 1:
        return ["score"]
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, Mapping):
        return [str(id2label.get(index, f"score_{index}")) for index in range(num_channels)]
    return [f"score_{index}" for index in range(num_channels)]


def _score_value(scores: Tensor) -> float | list[float]:
    values = scores.reshape(-1).tolist()
    if len(values) == 1:
        return float(values[0])
    return [float(value) for value in values]


__all__ = ["SpliceSitePredictionPipeline", "SpliceVariantEffectPipeline"]
