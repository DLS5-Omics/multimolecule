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
from inspect import signature
from typing import Any, Mapping

import torch
from torch import Tensor
from transformers.pipelines.base import GenericTensor, Pipeline, PipelineException


class RegulatoryActivityPipeline(Pipeline):
    """
    Regulatory activity pipeline for DNA regulatory models.

    The pipeline accepts raw nucleotide sequences and returns whole-sequence regulatory scores. Models with auxiliary
    numeric features, such as Xpresso, can receive them through the optional `features=` argument.
    """

    def preprocess(
        self,
        inputs: str | Mapping[str, Any],
        features: Any | None = None,
        return_tensors: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **preprocess_parameters,
    ) -> dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = "pt"
        sequence, features = _resolve_sequence_inputs(inputs, features)
        tokenizer_kwargs = _tokenizer_kwargs(tokenizer_kwargs)
        model_inputs = self.tokenizer(sequence, return_tensors=return_tensors, **tokenizer_kwargs)
        if features is not None:
            model_inputs["features"] = _prepare_features(features)
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
        config = getattr(self.model, "config", None)
        scores, channels = _processed_scores(
            model_outputs, model=self.model, config=config, sequence_level=True, task=self.task
        )

        results = [
            _sequence_prediction_result(sequence, sample_scores, channels=channels)
            for sequence, sample_scores in zip(sequences, _sample_tensors(scores, len(sequences), token_level=False))
        ]
        if len(results) == 1:
            return results[0]
        return results

    def _sanitize_parameters(
        self,
        features: Any | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        preprocess_params: dict[str, Any] = {}
        if features is not None:
            preprocess_params["features"] = features
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs
        return preprocess_params, {}, {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            raise NotImplementedError("Only PyTorch is supported for regulatory activity.")


class RegulatoryTrackPipeline(Pipeline):
    """
    Binned regulatory track pipeline for DNA regulatory models.

    The pipeline accepts raw nucleotide sequences and returns per-bin regulatory track scores, matching models such as
    Basenji and Enformer.
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
        sequence, _ = _resolve_sequence_inputs(inputs, None)
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
        config = getattr(self.model, "config", None)
        scores, channels = _processed_scores(
            model_outputs, model=self.model, config=config, sequence_level=False, task=self.task
        )

        results = [
            _track_prediction_result(sequence, sample_scores, channels=channels, config=config)
            for sequence, sample_scores in zip(sequences, _sample_tensors(scores, len(sequences), token_level=True))
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
            raise NotImplementedError("Only PyTorch is supported for regulatory track prediction.")


class RegulatoryProfilePipeline(Pipeline):
    """
    Base-resolution regulatory profile pipeline for DNA regulatory models.

    The pipeline accepts raw nucleotide sequences and returns per-base regulatory profiles. Models with factorized
    profile/count heads can provide a `postprocess` method to recombine those outputs for inference.
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
        sequence, _ = _resolve_sequence_inputs(inputs, None)
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
        config = getattr(self.model, "config", None)
        scores, channels = _processed_scores(
            model_outputs, model=self.model, config=config, sequence_level=False, task=self.task
        )

        results = [
            _profile_prediction_result(sequence, sample_scores, channels=channels)
            for sequence, sample_scores in zip(sequences, _sample_tensors(scores, len(sequences), token_level=True))
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
            raise NotImplementedError("Only PyTorch is supported for regulatory profile prediction.")


class RegulatoryVariantEffectPipeline(Pipeline):
    """
    Regulatory variant-effect pipeline.

    The pipeline scores a reference DNA sequence and an alternative DNA sequence and returns the
    alternative-minus-reference delta. Models with a native paired-input variant-effect head are called once; other
    models are scored on reference and alternative separately.
    """

    top_k: int | None = 20

    def preprocess(
        self,
        inputs: str | Mapping[str, Any],
        alternative: str | None = None,
        features: Any | None = None,
        alternative_features: Any | None = None,
        return_tensors: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **preprocess_parameters,
    ) -> dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = "pt"
        reference, alternative, features, alternative_features = _resolve_variant_inputs(
            inputs, alternative, features, alternative_features
        )
        tokenizer_kwargs = _tokenizer_kwargs(tokenizer_kwargs)
        reference_inputs = self.tokenizer(reference, return_tensors=return_tensors, **tokenizer_kwargs)
        alternative_inputs = self.tokenizer(alternative, return_tensors=return_tensors, **tokenizer_kwargs)
        output = {
            **reference_inputs,
            "reference_sequence": reference,
            "alternative_sequence": alternative,
            "alternative_input_ids": alternative_inputs["input_ids"],
            "alternative_attention_mask": alternative_inputs.get("attention_mask"),
        }
        if features is not None:
            output["features"] = _prepare_features(features)
        if alternative_features is not None:
            output["alternative_features"] = _prepare_features(alternative_features)
        return output

    def _forward(self, model_inputs):
        reference_sequence = model_inputs.pop("reference_sequence")
        alternative_sequence = model_inputs.pop("alternative_sequence")
        alternative_input_ids = model_inputs.pop("alternative_input_ids")
        alternative_attention_mask = model_inputs.pop("alternative_attention_mask", None)
        reference_features = model_inputs.pop("features", None)
        alternative_features = model_inputs.pop("alternative_features", reference_features)

        if _model_accepts_argument(self.model, "alternative_input_ids"):
            paired_inputs = {
                **model_inputs,
                "alternative_input_ids": alternative_input_ids,
                "alternative_attention_mask": alternative_attention_mask,
            }
            if reference_features is not None:
                paired_inputs["features"] = reference_features
            if alternative_features is not None and _model_accepts_argument(self.model, "alternative_features"):
                paired_inputs["alternative_features"] = alternative_features
            outputs = _call_model(self.model, paired_inputs)
            outputs["reference_sequence"] = reference_sequence
            outputs["alternative_sequence"] = alternative_sequence
            return outputs

        reference_outputs = _call_model(self.model, {**model_inputs, "features": reference_features})
        alternative_outputs = _call_model(
            self.model,
            {
                "input_ids": alternative_input_ids,
                "attention_mask": alternative_attention_mask,
                "features": alternative_features,
            },
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
                model_outputs["reference_outputs"],
                model=self.model,
                config=config,
                sequence_level=True,
                task=self.task,
            )
            alternative_scores, _ = _processed_scores(
                model_outputs["alternative_outputs"],
                model=self.model,
                config=config,
                sequence_level=True,
                task=self.task,
            )
            delta_scores = alternative_scores - reference_scores
            return _variant_effect_result(
                reference_sequence,
                alternative_sequence,
                delta_scores,
                channels,
                reference_scores=reference_scores,
                alternative_scores=alternative_scores,
                axis_name=_variant_axis_name(delta_scores, config),
                top_k=top_k,
            )

        delta_scores, channels = _processed_scores(
            model_outputs,
            model=self.model,
            config=config,
            sequence_level=True,
            task=self.task,
        )
        return _variant_effect_result(
            reference_sequence,
            alternative_sequence,
            delta_scores,
            channels,
            axis_name=_variant_axis_name(delta_scores, config),
            top_k=top_k,
        )

    def _sanitize_parameters(
        self,
        alternative: str | None = None,
        features: Any | None = None,
        alternative_features: Any | None = None,
        top_k: int | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        preprocess_params: dict[str, Any] = {}
        if alternative is not None:
            preprocess_params["alternative"] = alternative
        if features is not None:
            preprocess_params["features"] = features
        if alternative_features is not None:
            preprocess_params["alternative_features"] = alternative_features
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
            raise NotImplementedError("Only PyTorch is supported for regulatory variant-effect prediction.")

    def __call__(self, inputs, alternative: str | None = None, **kwargs):
        if alternative is not None:
            kwargs["alternative"] = alternative
        return super().__call__(inputs, **kwargs)


def _sequence_prediction_result(sequence: str, scores: Tensor, *, channels: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"sequence": sequence, "channels": channels}
    if scores.ndim == 1:
        if len(channels) == 1:
            result["score"] = _score_value(scores)
        else:
            result["scores"] = _channel_scores(scores, channels)
        return result
    result["position_index_base"] = 0
    result["scores"] = _axis_scores(scores, channels, "position", sequence=sequence)
    return result


def _track_prediction_result(sequence: str, scores: Tensor, *, channels: list[str], config: Any) -> dict[str, Any]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(-1)
    result = {
        "sequence": sequence,
        "bin_index_base": 0,
        "tracks": _axis_scores(scores, channels, "bin"),
        "channels": channels,
    }
    result.update(_bin_metadata(config))
    return result


def _profile_prediction_result(sequence: str, scores: Tensor, *, channels: list[str]) -> dict[str, Any]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(-1)
    length = min(len(sequence), scores.shape[0])
    return {
        "sequence": sequence,
        "position_index_base": 0,
        "profile": _axis_scores(scores[:length], channels, "position", sequence=sequence[:length]),
        "channels": channels,
    }


def _variant_effect_result(
    reference_sequence: str,
    alternative_sequence: str,
    delta_scores: Tensor,
    channels: list[str],
    *,
    reference_scores: Tensor | None = None,
    alternative_scores: Tensor | None = None,
    axis_name: str = "position",
    top_k: int | None = 20,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_sequence": reference_sequence,
        "alternative_sequence": alternative_sequence,
        "channels": channels,
    }
    if delta_scores.ndim == 1:
        if len(channels) == 1:
            result["delta_score"] = _score_value(delta_scores)
            if reference_scores is not None:
                result["reference_score"] = _score_value(reference_scores)
            if alternative_scores is not None:
                result["alternative_score"] = _score_value(alternative_scores)
            result["variant_effects"] = _channel_delta_effects(
                delta_scores,
                channels,
                reference_scores=reference_scores,
                alternative_scores=alternative_scores,
                top_k=top_k,
            )
            return result
        result["delta_scores"] = _channel_scores(delta_scores, channels)
        if reference_scores is not None:
            result["reference_scores"] = _channel_scores(reference_scores, channels)
        if alternative_scores is not None:
            result["alternative_scores"] = _channel_scores(alternative_scores, channels)
        result["variant_effects"] = _channel_delta_effects(
            delta_scores,
            channels,
            reference_scores=reference_scores,
            alternative_scores=alternative_scores,
            top_k=top_k,
        )
        return result

    length = min(len(reference_sequence), delta_scores.shape[0]) if axis_name == "position" else delta_scores.shape[0]
    result[f"{axis_name}_index_base"] = 0
    axis_sequence = reference_sequence if axis_name == "position" else None
    result["delta_scores"] = _axis_scores(delta_scores[:length], channels, axis_name, sequence=axis_sequence)
    if reference_scores is not None:
        result["reference_scores"] = _axis_scores(
            reference_scores[:length], channels, axis_name, sequence=axis_sequence
        )
    if alternative_scores is not None:
        alternative_axis_sequence = alternative_sequence if axis_name == "position" else None
        result["alternative_scores"] = _axis_scores(
            alternative_scores[:length], channels, axis_name, sequence=alternative_axis_sequence
        )
    result["variant_effects"] = _axis_delta_effects(
        reference_sequence[:length],
        alternative_sequence[:length],
        delta_scores[:length],
        channels,
        axis_name=axis_name,
        reference_scores=reference_scores[:length] if reference_scores is not None else None,
        alternative_scores=alternative_scores[:length] if alternative_scores is not None else None,
        top_k=top_k,
    )
    return result


def _processed_scores(
    model_outputs: Any,
    *,
    model: Any,
    config: Any,
    sequence_level: bool,
    task: str,
) -> tuple[Tensor, list[str]]:
    channels = None
    postprocess = getattr(model, "postprocess", None)
    if callable(postprocess):
        processed = postprocess(model_outputs)
        if isinstance(processed, tuple) and len(processed) == 2:
            scores, channels = processed
        else:
            scores = processed
    else:
        scores = _get_logits(model_outputs, task=task)

    if not isinstance(scores, Tensor):
        scores = torch.as_tensor(scores)
    scores = scores.detach().float().cpu()
    if sequence_level and scores.ndim > 1 and scores.shape[0] == 1:
        scores = scores.squeeze(0)
    if scores.ndim == 0:
        scores = scores.unsqueeze(0)
    if scores.ndim == 1:
        num_channels = scores.shape[0] if sequence_level else 1
    else:
        num_channels = scores.shape[-1]
    if channels is None:
        channels = _sequence_channels(num_channels, model, config)
    else:
        channels = list(channels)
    return scores, channels


def _axis_scores(
    scores: Tensor,
    channels: list[str],
    axis_name: str,
    *,
    sequence: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in range(scores.shape[0]):
        row: dict[str, Any] = {axis_name: position}
        if sequence is not None:
            row["nucleotide"] = sequence[position] if position < len(sequence) else None
        row.update({channel: float(scores[position, index].item()) for index, channel in enumerate(channels)})
        rows.append(row)
    return rows


def _channel_scores(scores: Tensor, channels: list[str]) -> dict[str, float]:
    values = scores.reshape(-1).tolist()
    return {channel: float(values[index]) for index, channel in enumerate(channels)}


def _score_value(scores: Tensor) -> float | list[float]:
    values = scores.reshape(-1).tolist()
    if len(values) == 1:
        return float(values[0])
    return [float(value) for value in values]


def _get_logits(outputs: Any, *, task: str) -> Tensor:
    logits = outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", None)
    if logits is None:
        raise PipelineException(task, "model", "Unable to find logits in model outputs.")
    if not isinstance(logits, Tensor):
        logits = torch.as_tensor(logits)
    return logits


def _sample_tensors(scores: Tensor, sample_count: int, *, token_level: bool) -> list[Tensor]:
    if sample_count == 1:
        if token_level and scores.ndim <= 2:
            return [scores]
        if not token_level and scores.ndim <= 1:
            return [scores]
    return list(scores)


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


def _resolve_sequence_inputs(inputs: str | Mapping[str, Any], features: Any | None) -> tuple[str, Any | None]:
    if isinstance(inputs, Mapping):
        sequence = inputs.get("sequence") or inputs.get("reference") or inputs.get("reference_sequence")
        if features is None:
            features = inputs.get("features")
    else:
        sequence = inputs
    if not sequence:
        raise ValueError("regulatory prediction requires a DNA sequence.")
    return str(sequence), features


def _resolve_variant_inputs(
    inputs: str | Mapping[str, Any],
    alternative: str | None,
    features: Any | None,
    alternative_features: Any | None,
) -> tuple[str, str, Any | None, Any | None]:
    if isinstance(inputs, Mapping):
        reference = inputs.get("reference") or inputs.get("reference_sequence") or inputs.get("sequence")
        alternative = alternative or inputs.get("alternative") or inputs.get("alternative_sequence")
        if features is None:
            features = inputs.get("features")
        if alternative_features is None:
            alternative_features = inputs.get("alternative_features")
    else:
        reference = inputs
    if not reference:
        raise ValueError("regulatory-variant-effect requires a reference sequence.")
    if not alternative:
        raise ValueError("regulatory-variant-effect requires an alternative sequence.")
    if len(reference) != len(alternative):
        raise ValueError(
            "regulatory-variant-effect currently requires reference and alternative sequences with the same length."
        )
    return str(reference), str(alternative), features, alternative_features


def _prepare_features(features: Any) -> Tensor:
    if isinstance(features, Tensor):
        tensor = features
    else:
        tensor = torch.as_tensor(features, dtype=torch.float32)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _call_model(model, model_inputs: Mapping[str, Any]):
    inputs = dict(model_inputs)
    features = inputs.pop("features", None)
    if features is not None and _model_accepts_argument(model, "features"):
        inputs["features"] = features
    return model(**{key: value for key, value in inputs.items() if value is not None})


def _model_accepts_argument(model, argument: str) -> bool:
    try:
        parameters = signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return argument in parameters


def _tokenizer_kwargs(tokenizer_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = {"add_special_tokens": False}
    if tokenizer_kwargs is not None:
        kwargs.update(tokenizer_kwargs)
    return kwargs


def _channel_delta_effects(
    delta_scores: Tensor,
    channels: list[str],
    *,
    reference_scores: Tensor | None,
    alternative_scores: Tensor | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    effects = []
    flat_delta = delta_scores.reshape(-1)
    flat_reference = reference_scores.reshape(-1) if reference_scores is not None else None
    flat_alternative = alternative_scores.reshape(-1) if alternative_scores is not None else None
    for channel_index, channel in enumerate(channels[: flat_delta.shape[0]]):
        delta_score = _finite_float(flat_delta[channel_index])
        if delta_score is None:
            continue
        effect: dict[str, Any] = {
            "channel": channel,
            "delta_score": delta_score,
            "direction": _delta_direction(delta_score),
        }
        if flat_reference is not None:
            effect["reference_score"] = _finite_float(flat_reference[channel_index])
        if flat_alternative is not None:
            effect["alternative_score"] = _finite_float(flat_alternative[channel_index])
        effects.append(effect)
    return _rank_delta_effects(effects, top_k)


def _axis_delta_effects(
    reference_sequence: str,
    alternative_sequence: str,
    delta_scores: Tensor,
    channels: list[str],
    *,
    axis_name: str,
    reference_scores: Tensor | None,
    alternative_scores: Tensor | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    effects = []
    for axis_index in range(delta_scores.shape[0]):
        for channel_index, channel in enumerate(channels[: delta_scores.shape[-1]]):
            delta_score = _finite_float(delta_scores[axis_index, channel_index])
            if delta_score is None:
                continue
            effect: dict[str, Any] = {
                axis_name: axis_index,
                "channel": channel,
                "delta_score": delta_score,
                "direction": _delta_direction(delta_score),
            }
            if axis_name == "position":
                effect["reference_nucleotide"] = (
                    reference_sequence[axis_index] if axis_index < len(reference_sequence) else None
                )
                effect["alternative_nucleotide"] = (
                    alternative_sequence[axis_index] if axis_index < len(alternative_sequence) else None
                )
            if reference_scores is not None:
                effect["reference_score"] = _finite_float(reference_scores[axis_index, channel_index])
            if alternative_scores is not None:
                effect["alternative_score"] = _finite_float(alternative_scores[axis_index, channel_index])
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


def _bin_metadata(config: Any) -> dict[str, int]:
    pool_factor = getattr(config, "pool_factor", None)
    if isinstance(pool_factor, int):
        return {"bin_size": pool_factor, "bin_stride": pool_factor}
    return {}


def _variant_axis_name(scores: Tensor, config: Any) -> str:
    if scores.ndim > 1 and isinstance(getattr(config, "pool_factor", None), int):
        return "bin"
    return "position"


def _sequence_channels(num_channels: int, model: Any, config: Any) -> list[str]:
    output_channels = getattr(model, "output_channels", None)
    if callable(output_channels):
        output_channels = output_channels()
    if output_channels is not None:
        channels = [str(channel) for channel in output_channels]
        if len(channels) == num_channels:
            return channels
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, Mapping):
        return [str(id2label.get(index, f"score_{index}")) for index in range(num_channels)]
    if num_channels == 1:
        return ["score"]
    return [f"score_{index}" for index in range(num_channels)]


def _model_prefix(model: Any) -> str:
    return getattr(model, "base_model_prefix", "model")


__all__ = [
    "RegulatoryActivityPipeline",
    "RegulatoryTrackPipeline",
    "RegulatoryProfilePipeline",
    "RegulatoryVariantEffectPipeline",
]
