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
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import can_return_tuple

from multimolecule.modules import TokenPredictionHead, preserve_batch_norm_stats

from .configuration_deltasplice import DeltaSpliceConfig


class DeltaSplicePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeltaSpliceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeltaSpliceLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.BatchNorm1d) and module.affine:
            init.ones_(module.weight)
            init.zeros_(module.bias)


class DeltaSpliceModel(DeltaSplicePreTrainedModel):
    """
    DeltaSplice backbone and official five-seed ensemble for per-position splice-site usage prediction.

    Examples:
        >>> import torch
        >>> from multimolecule.models.deltasplice import DeltaSpliceConfig, DeltaSpliceLayerConfig, DeltaSpliceModel
        >>> layer = DeltaSpliceLayerConfig(kernel_size=3, dilation=1)
        >>> config = DeltaSpliceConfig(context=4, hidden_size=8, layers=[layer], num_ensemble=1)
        >>> model = DeltaSpliceModel(config)
        >>> output = model(torch.randint(5, (1, 6)))
        >>> output["probabilities"].shape
        torch.Size([1, 6, 3])
    """

    def __init__(self, config: DeltaSpliceConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = DeltaSpliceEmbedding(config)
        self.members = nn.ModuleList([DeltaSpliceModule(config) for _ in range(config.num_ensemble)])

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_labels == 3:
            return ["no_splice", "acceptor", "donor"]
        return [f"label_{index}" for index in range(self.config.num_labels)]

    def postprocess(self, outputs: DeltaSpliceModelOutput | ModelOutput | Tensor) -> tuple[Tensor, list[str]]:
        r"""
        Return DeltaSplice splice-site usage probabilities with semantic channel names.

        Args:
            outputs: The output of [`DeltaSpliceModel`][multimolecule.models.DeltaSpliceModel], or its
                `probabilities` tensor.

        Returns:
            A tuple of `(scores, channels)`, where `scores` are splice-site usage probabilities, or the probability
            change (`delta`) when an alternative sequence is scored.
        """
        if isinstance(outputs, Tensor):
            return outputs, self.output_channels
        scores = outputs["delta"] if outputs.get("delta") is not None else outputs["probabilities"]
        return scores, self.output_channels

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        alternative_input_ids: Tensor | NestedTensor | None = None,
        alternative_attention_mask: Tensor | None = None,
        alternative_inputs_embeds: Tensor | NestedTensor | None = None,
        reference_input_ids: Tensor | NestedTensor | None = None,
        reference_attention_mask: Tensor | None = None,
        reference_inputs_embeds: Tensor | NestedTensor | None = None,
        reference_usage: Tensor | None = None,
        use_reference: bool | None = None,
        output_contexts: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> DeltaSpliceModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if alternative_input_ids is not None and alternative_inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both alternative_input_ids and alternative_inputs_embeds at the same time"
            )
        if reference_input_ids is not None and reference_inputs_embeds is not None:
            raise ValueError("You cannot specify both reference_input_ids and reference_inputs_embeds at the same time")

        output_contexts = output_contexts if output_contexts is not None else self.config.output_contexts
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        record_contexts = bool(output_contexts) or bool(output_hidden_states)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        alternative_embedding_output = None
        if alternative_input_ids is not None or alternative_inputs_embeds is not None:
            alternative_embedding_output = self.embeddings(
                input_ids=alternative_input_ids,
                attention_mask=alternative_attention_mask,
                inputs_embeds=alternative_inputs_embeds,
            )
        reference_embedding_output = None
        if reference_input_ids is not None or reference_inputs_embeds is not None:
            reference_embedding_output = self.embeddings(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                inputs_embeds=reference_inputs_embeds,
            )

        if alternative_embedding_output is not None and reference_embedding_output is not None:
            raise ValueError("Use either alternative_* inputs or reference_* inputs in one DeltaSplice call, not both.")

        if alternative_embedding_output is None:
            member_outputs = [
                member(
                    embedding_output,
                    reference_embeds=reference_embedding_output,
                    reference_usage=reference_usage,
                    output_contexts=record_contexts,
                    output_hidden_states=record_contexts,
                )
                for member in self.members
            ]
        else:
            member_outputs = [
                _member_variant_output(
                    member,
                    embedding_output,
                    alternative_embedding_output,
                    reference_usage=reference_usage,
                    use_reference=bool(use_reference) or reference_usage is not None,
                    output_contexts=record_contexts,
                    output_hidden_states=record_contexts,
                )
                for member in self.members
            ]

        last_hidden_state = _average_tensors([out.last_hidden_state for out in member_outputs])
        probabilities = _average_tensors([out.probabilities for out in member_outputs])
        site_probabilities = _average_tensors([out.site_probabilities for out in member_outputs])
        alternative_probabilities = _average_tensors([out.alternative_probabilities for out in member_outputs])
        delta = _average_tensors([out.delta for out in member_outputs])

        contexts: tuple[Tensor, ...] | None = None
        if record_contexts:
            per_member_contexts = [out.contexts for out in member_outputs if out.contexts is not None]
            if per_member_contexts:
                num_contexts = len(per_member_contexts[0])
                contexts = tuple(
                    _average_tensors([member_contexts[index] for member_contexts in per_member_contexts])
                    for index in range(num_contexts)
                )

        return DeltaSpliceModelOutput(
            last_hidden_state=last_hidden_state,
            probabilities=probabilities,
            site_probabilities=site_probabilities,
            alternative_probabilities=alternative_probabilities,
            delta=delta,
            contexts=contexts if output_contexts else None,
            hidden_states=contexts if output_hidden_states else None,
        )


class DeltaSpliceForTokenPrediction(DeltaSplicePreTrainedModel):
    """
    DeltaSplice model with a shared MultiMolecule token prediction head.

    Examples:
        >>> import torch
        >>> from multimolecule.models.deltasplice import (
        ...     DeltaSpliceConfig,
        ...     DeltaSpliceForTokenPrediction,
        ...     DeltaSpliceLayerConfig,
        ... )
        >>> layer = DeltaSpliceLayerConfig(kernel_size=3, dilation=1)
        >>> config = DeltaSpliceConfig(context=4, hidden_size=8, layers=[layer], num_ensemble=1)
        >>> model = DeltaSpliceForTokenPrediction(config)
        >>> output = model(torch.randint(5, (1, 6)), labels=torch.rand(1, 6, 3))
        >>> output["logits"].shape
        torch.Size([1, 6, 3])
    """

    def __init__(self, config: DeltaSpliceConfig):
        super().__init__(config)
        self.model = DeltaSpliceModel(config)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        reference_input_ids: Tensor | NestedTensor | None = None,
        reference_attention_mask: Tensor | None = None,
        reference_inputs_embeds: Tensor | NestedTensor | None = None,
        reference_usage: Tensor | None = None,
        labels: Tensor | None = None,
        output_contexts: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> DeltaSpliceTokenPredictorOutput:
        head_attention_mask = attention_mask
        if input_ids is None and inputs_embeds is not None and head_attention_mask is None:
            if isinstance(inputs_embeds, NestedTensor):
                head_attention_mask = inputs_embeds.mask
            else:
                head_attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.int, device=inputs_embeds.device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            reference_input_ids=reference_input_ids,
            reference_attention_mask=reference_attention_mask,
            reference_inputs_embeds=reference_inputs_embeds,
            reference_usage=reference_usage,
            output_contexts=output_contexts,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        output = self.token_head(outputs, head_attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return DeltaSpliceTokenPredictorOutput(
            loss=loss,
            logits=logits,
            contexts=outputs.contexts,
            hidden_states=outputs.hidden_states,
        )


class DeltaSpliceModule(nn.Module):
    """A single DeltaSplice seed checkpoint member."""

    def __init__(self, config: DeltaSpliceConfig):
        super().__init__()
        self.encoder = DeltaSpliceEncoder(config)
        self.reference_projection = DeltaSpliceReferenceProjection(config)
        self.usage_prediction = DeltaSplicePredictionHead(config)
        self.delta_prediction = DeltaSplicePredictionHead(config)
        self.site_prediction = DeltaSplicePredictionHead(config)

    def forward(
        self,
        inputs_embeds: Tensor,
        reference_embeds: Tensor | None = None,
        reference_usage: Tensor | None = None,
        output_contexts: bool = False,
        output_hidden_states: bool = False,
    ) -> DeltaSpliceModuleOutput:
        encoder_outputs = self.encoder(
            inputs_embeds,
            output_contexts=output_contexts,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = encoder_outputs.last_hidden_state
        site_probabilities = self.site_prediction(hidden_state)

        if reference_embeds is None:
            probabilities = self.usage_prediction(hidden_state)
        else:
            reference_outputs = self.encoder(reference_embeds)
            reference_hidden_state = reference_outputs.last_hidden_state
            if reference_usage is None or bool(torch.isnan(reference_usage.sum()).item()):
                reference_usage = self.usage_prediction(reference_hidden_state)
            else:
                reference_usage = reference_usage.to(device=hidden_state.device, dtype=hidden_state.dtype)
            reference_features = self.reference_projection(hidden_state, reference_hidden_state, reference_usage)
            probabilities = self.delta_prediction(hidden_state - reference_hidden_state + reference_features)

        return DeltaSpliceModuleOutput(
            probabilities=probabilities,
            site_probabilities=site_probabilities,
            last_hidden_state=hidden_state,
            contexts=encoder_outputs.contexts,
            hidden_states=encoder_outputs.hidden_states,
        )


def _member_variant_output(
    member: DeltaSpliceModule,
    reference_embeds: Tensor,
    alternative_embeds: Tensor,
    reference_usage: Tensor | None,
    use_reference: bool,
    output_contexts: bool,
    output_hidden_states: bool,
) -> DeltaSpliceModuleOutput:
    reference = member(
        reference_embeds,
        output_contexts=output_contexts,
        output_hidden_states=output_hidden_states,
    )
    if use_reference:
        usage = reference.probabilities if reference_usage is None else reference_usage
        reference_scored = member(reference_embeds, reference_embeds=reference_embeds, reference_usage=usage)
        alternative = member(alternative_embeds, reference_embeds=reference_embeds, reference_usage=usage)
        probabilities = reference_scored.probabilities
    else:
        alternative = member(alternative_embeds)
        probabilities = reference.probabilities
    return DeltaSpliceModuleOutput(
        probabilities=probabilities,
        site_probabilities=reference.site_probabilities,
        last_hidden_state=reference.last_hidden_state,
        alternative_probabilities=alternative.probabilities,
        delta=alternative.probabilities - probabilities,
        contexts=reference.contexts,
        hidden_states=reference.hidden_states,
    )


class DeltaSpliceEmbedding(nn.Module):
    """One-hot encode nucleotide tokens and zero-pad DeltaSplice's flanking context."""

    def __init__(self, config: DeltaSpliceConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_tokens = config.vocab_size + 1
        self.padding = config.context // 2
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = F.one_hot(input_ids, num_classes=self.num_tokens)[..., : self.vocab_size].to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        if self.padding > 0:
            batch_size = inputs_embeds.size(0)
            padding = torch.zeros(
                batch_size, self.vocab_size, self.padding, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = torch.cat([padding, inputs_embeds, padding], dim=2)
        return inputs_embeds


class DeltaSpliceEncoder(nn.Module):
    def __init__(self, config: DeltaSpliceConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Conv1d(config.vocab_size, config.hidden_size, 1)
        self.layers = nn.ModuleList(
            DeltaSpliceLayer(config, layer.kernel_size, layer.dilation) for layer in config.layers
        )
        self.norm = nn.BatchNorm1d(
            config.hidden_size,
            eps=config.batch_norm_eps,
            momentum=config.batch_norm_momentum,
            affine=False,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        output_contexts: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> DeltaSpliceEncoderOutput:
        target_length = hidden_state.size(2) - self.config.context
        if target_length <= 0:
            raise ValueError(
                f"DeltaSplice input length after padding ({hidden_state.size(2)}) must exceed context "
                f"({self.config.context})."
            )

        record_contexts = bool(output_contexts) or bool(output_hidden_states)
        all_contexts: tuple[Tensor, ...] | None = () if record_contexts else None

        hidden_state = self.projection(hidden_state)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_state,
                    context_fn=lambda layer=layer: (nullcontext(), preserve_batch_norm_stats(layer)),
                )
            else:
                hidden_state = layer(hidden_state)
            if record_contexts:
                context = _center_crop(hidden_state, target_length).transpose(1, 2)
                all_contexts = all_contexts + (context,)  # type: ignore[operator]

        hidden_state = self.norm(hidden_state)
        hidden_state = _center_crop(hidden_state, target_length).transpose(1, 2)

        return DeltaSpliceEncoderOutput(
            last_hidden_state=hidden_state,
            contexts=all_contexts if output_contexts else None,
            hidden_states=all_contexts if output_hidden_states else None,
        )


class DeltaSpliceLayer(nn.Module):
    def __init__(self, config: DeltaSpliceConfig, kernel_size: int, dilation: int = 1):
        super().__init__()
        if isinstance(config.hidden_act, str):
            act1 = ACT2FN[config.hidden_act]
            act2 = ACT2FN[config.hidden_act]
        else:
            act1 = config.hidden_act
            act2 = config.hidden_act
        self.norm1 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = act1
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act2 = act2
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=kernel_size, dilation=dilation)

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.conv2(hidden_state)
        residual = _center_crop(residual, hidden_state.size(-1))
        return hidden_state + residual


class DeltaSpliceReferenceProjection(nn.Module):
    def __init__(self, config: DeltaSpliceConfig):
        super().__init__()
        self.dense = nn.Linear(2 + config.hidden_size * 2, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, config.hidden_size)

    def forward(self, target_feature: Tensor, reference_feature: Tensor, reference_usage: Tensor) -> Tensor:
        usage = torch.log(reference_usage[..., 1:].clamp(1e-10))
        hidden_state = torch.cat([usage, target_feature, reference_feature], dim=-1)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.act(hidden_state)
        return self.output(hidden_state)


class DeltaSplicePredictionHead(nn.Module):
    def __init__(self, config: DeltaSpliceConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 256)
        self.act1 = nn.ReLU()
        self.intermediate = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(256, config.num_labels)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.intermediate(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.output(hidden_state)
        return F.softmax(hidden_state, dim=-1)


def _center_crop(hidden_state: Tensor, target_length: int) -> Tensor:
    length = hidden_state.size(-1)
    if length == target_length:
        return hidden_state
    if length < target_length:
        raise ValueError(f"Cannot center-crop length {length} to larger target length {target_length}.")
    crop = length - target_length
    if crop % 2:
        raise ValueError(f"DeltaSplice valid-convolution crop must be even, got {crop}.")
    half = crop // 2
    return hidden_state[..., half:-half]


def _average_tensors(values: list[Tensor | None]) -> Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


@dataclass
class DeltaSpliceModelOutput(ModelOutput):
    """
    Base class for outputs of the DeltaSplice model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Per-position encoder representation averaged across ensemble members.
        probabilities (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            DeltaSplice splice-site usage probabilities (`no_splice`, `acceptor`, `donor`) averaged across ensemble
            members. These are softmax-normalised; DeltaSplice has no pre-softmax logit surface.
        site_probabilities (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            DeltaSplice splice-site probability module outputs averaged across ensemble members.
        alternative_probabilities
            (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`, *optional*):
            Alternative-sequence splice-site usage probabilities when an alternative sequence is supplied.
        delta (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`, *optional*):
            `alternative_probabilities - probabilities` when an alternative sequence is supplied.
        contexts (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer valid-convolution representations cropped to the input sequence length.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Same content as `contexts`; provided for Transformers hidden-state compatibility.
    """

    last_hidden_state: torch.FloatTensor | None = None
    probabilities: torch.FloatTensor | None = None
    site_probabilities: torch.FloatTensor | None = None
    alternative_probabilities: torch.FloatTensor | None = None
    delta: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class DeltaSpliceEncoderOutput(ModelOutput):
    """Internal output of a single DeltaSplice encoder (one ensemble member's convolutional stack)."""

    last_hidden_state: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class DeltaSpliceModuleOutput(ModelOutput):
    """Internal output of a single DeltaSplice ensemble member, holding softmax-normalised probabilities."""

    probabilities: torch.FloatTensor | None = None
    site_probabilities: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    alternative_probabilities: torch.FloatTensor | None = None
    delta: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class DeltaSpliceTokenPredictorOutput(ModelOutput):
    """
    Base class for outputs of DeltaSplice token prediction models.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Token prediction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            Per-nucleotide token prediction outputs.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True`):
            Per-layer context representations.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Per-layer context representations.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
