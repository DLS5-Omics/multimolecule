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
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import preserve_batch_norm_stats

from .configuration_spliceai import SpliceAiConfig


class SpliceAiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpliceAiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["SpliceAiBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class SpliceAiModel(SpliceAiPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import SpliceAiConfig, SpliceAiModel, RnaTokenizer
        >>> config = SpliceAiConfig()
        >>> model = SpliceAiModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/spliceai")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits"].shape
        torch.Size([1, 5, 3])
    """

    def __init__(self, config: SpliceAiConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.embeddings = SpliceAiEmbedding(config)
        # The original Illumina SpliceAI release ships an ensemble of 5 identically-shaped networks.
        self.members = nn.ModuleList([SpliceAiModule(config) for _ in range(5)])
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SpliceAiModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        output_contexts = kwargs.get("output_contexts", self.config.output_contexts)
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        record_contexts = bool(output_contexts) or bool(output_hidden_states)
        kwargs["output_contexts"] = record_contexts
        kwargs["output_hidden_states"] = record_contexts

        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        member_outputs = [member(embedding_output, labels=labels, **kwargs) for member in self.members]

        logits = _average_tensors([out.logits for out in member_outputs])
        losses = [out.loss for out in member_outputs if out.loss is not None]
        loss = torch.stack(losses).mean() if losses else None

        contexts: tuple[Tensor, ...] | None = None
        if record_contexts:
            per_member_contexts = [out.contexts for out in member_outputs if out.contexts is not None]
            if per_member_contexts:
                num_stages = len(per_member_contexts[0])
                contexts = tuple(_average_tensors([m[idx] for m in per_member_contexts]) for idx in range(num_stages))

        return SpliceAiModelOutput(
            loss=loss,
            logits=logits,
            contexts=contexts if output_contexts else None,
            hidden_states=contexts if output_hidden_states else None,
        )


class SpliceAiForTokenPrediction(SpliceAiModel):
    pass


class SpliceAiModule(nn.Module):
    def __init__(self, config: SpliceAiConfig):
        super().__init__()
        self.projection = nn.Conv1d(config.vocab_size, config.hidden_size, 1)
        self.encoder = SpliceAiEncoder(config)
        self.prediction = nn.Conv1d(config.hidden_size, config.num_labels, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: Tensor,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SpliceAiModuleOutput:
        embedding = self.projection(inputs_embeds)
        outputs = self.encoder(embedding, **kwargs)
        context = outputs.last_context
        logits = self.prediction(context)

        loss = self.criterion(logits, labels) if labels is not None else None
        logits = logits.transpose(1, 2)

        return SpliceAiModuleOutput(
            loss=loss,
            logits=logits,
            last_context=context,
            contexts=outputs.contexts,
        )


class SpliceAiEmbedding(nn.Module):
    def __init__(self, config: SpliceAiConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.context = config.context
        self.padding = config.context // 2
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if isinstance(input_ids, NestedTensor):
                storage = [F.one_hot(t, num_classes=self.vocab_size).to(dtype=dtype) for t in input_ids]
                inputs_embeds = _nested_like(input_ids, storage)
            else:
                inputs_embeds = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=dtype)
        elif isinstance(inputs_embeds, NestedTensor):
            inputs_embeds = _nested_like(inputs_embeds, [t.to(dtype=dtype) for t in inputs_embeds])
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None and not isinstance(inputs_embeds, NestedTensor):
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        if isinstance(inputs_embeds, NestedTensor):
            storage = []
            for t in inputs_embeds:
                pad = torch.zeros(t.size(0), self.padding, device=t.device, dtype=t.dtype)
                storage.append(torch.cat([pad, t, pad], dim=1))
            inputs_embeds = _nested_like(inputs_embeds, storage)
        else:
            batch_size = inputs_embeds.size(0)
            padding = torch.zeros(
                batch_size, self.vocab_size, self.padding, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = torch.cat([padding, inputs_embeds, padding], dim=2)
        return inputs_embeds


class SpliceAiEncoder(nn.Module):
    def __init__(self, config: SpliceAiConfig):
        super().__init__()
        self.config = config
        self.context = config.context
        self.padding = config.context // 2
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, 1, padding="same")
        self.stages = nn.ModuleList([SpliceAiStage(config, **s) for s in config.stages])
        self.context_recorder = SpliceAiContextRecorder()
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SpliceAiModuleOutput:
        record_contexts = kwargs.get("output_contexts", self.config.output_contexts) or kwargs.get(
            "output_hidden_states", self.config.output_hidden_states
        )
        all_contexts: tuple[Tensor, ...] | None = () if record_contexts else None
        context = self.conv(hidden_state)
        if record_contexts:
            trimmed_context = context[:, :, self.padding : -self.padding].transpose(1, 2)
            all_contexts = all_contexts + (self.context_recorder(trimmed_context),)  # type: ignore[operator]
        for stage in self.stages:
            if self.gradient_checkpointing and self.training:
                hidden_state, context = self._gradient_checkpointing_func(
                    stage.__call__,
                    hidden_state,
                    context,
                    context_fn=lambda stage=stage: (nullcontext(), preserve_batch_norm_stats(stage)),
                )
            else:
                hidden_state, context = stage(hidden_state, context)
            if record_contexts:
                trimmed_context = context[:, :, self.padding : -self.padding].transpose(1, 2)
                all_contexts = all_contexts + (self.context_recorder(trimmed_context),)  # type: ignore[operator]

        context = context[:, :, self.padding : -self.padding]
        return SpliceAiModuleOutput(last_context=context, contexts=all_contexts)


class SpliceAiStage(nn.Module):
    def __init__(self, config: SpliceAiConfig, num_blocks: int = 1, kernel_size=11, dilation: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SpliceAiBlock(config, kernel_size=kernel_size, dilation=dilation) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            padding="same",
        )

    def forward(self, hidden_state: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            hidden_state = block(hidden_state)
        context = self.conv(hidden_state) + context
        return hidden_state, context


class SpliceAiBlock(nn.Module):
    def __init__(self, config: SpliceAiConfig, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.norm2 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act2 = ACT2FN[config.hidden_act]
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class SpliceAiContextRecorder(nn.Module):
    def forward(self, context: Tensor) -> Tensor:
        return context


def _average_tensors(values: list[Tensor] | list[NestedTensor]) -> Tensor | NestedTensor:
    if isinstance(values[0], NestedTensor):
        stacked = torch.stack([v.tensor for v in values], dim=0)  # type: ignore[union-attr]
        mean = stacked.mean(dim=0)
        reference: NestedTensor = values[0]  # type: ignore[assignment]
        return NestedTensor.from_tensor_mask(
            mean,
            reference.mask,
            batch_first=reference.batch_first,
            padding_value=reference.padding_value,
            mask_value=reference.mask_value,
        )
    return torch.mean(torch.stack(values), dim=0)


def _nested_like(reference: NestedTensor, tensors: list[Tensor]) -> NestedTensor:
    """Build a NestedTensor that mirrors `reference`'s construction kwargs."""
    return NestedTensor(
        tensors,
        batch_first=reference.batch_first,
        padding_value=reference.padding_value,
        mask_value=reference.mask_value,
    )


@dataclass
class SpliceAiModelOutput(ModelOutput):
    """
    Base class for outputs of SpliceAI model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax), averaged across ensemble
            members.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True` is passed or when
            `config.output_contexts=True`):
            Tuple of `torch.FloatTensor` (one per stage of the encoder) of shape `(batch_size, sequence_length,
            hidden_size)`. Context vectors recorded after each stage, averaged across ensemble members.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one per stage of the encoder) of shape `(batch_size, sequence_length,
            hidden_size)`. Same content as `contexts`; provided for compatibility with the Transformers hidden-states
            convention.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class SpliceAiModuleOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_context: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


SpliceAiPreTrainedModel._can_record_outputs = {
    "contexts": SpliceAiContextRecorder,
}
