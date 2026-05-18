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
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import TokenPredictionHead, preserve_batch_norm_stats

from .configuration_pangolin import PangolinConfig


class PangolinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PangolinConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["PangolinBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
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


class PangolinModel(PangolinPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import PangolinConfig, PangolinModel, PangolinStageConfig
        >>> stage = PangolinStageConfig(num_blocks=1, kernel_size=3, dilation=1)
        >>> config = PangolinConfig(context=4, num_tissues=1, num_ensemble=1, stages=[stage])
        >>> model = PangolinModel(config)
        >>> input_ids = torch.randint(5, (1, 5))
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 5, 32])
        >>> output["probabilities"].shape
        torch.Size([1, 5, 3])
    """

    def __init__(self, config: PangolinConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = PangolinEmbedding(config)
        # The official Pangolin release uses one replicate ensemble per tissue output group.
        self.members = nn.ModuleList(
            [
                nn.ModuleList([PangolinModule(config) for _ in range(config.num_ensemble)])
                for _ in range(config.num_tissues)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PangolinModelOutput:
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
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        member_outputs = [
            [member(embedding_output, **kwargs) for member in tissue_members] for tissue_members in self.members
        ]
        flat_outputs = [output for tissue_outputs in member_outputs for output in tissue_outputs]

        last_hidden_state = _average_tensors([out.last_hidden_state for out in flat_outputs])
        tissue_probabilities = []
        for tissue, tissue_outputs in enumerate(member_outputs):
            probabilities = _average_tensors([out.probabilities for out in tissue_outputs])
            start = tissue * 3
            tissue_probabilities.append(probabilities[..., start : start + 3])
        probabilities = torch.cat(tissue_probabilities, dim=-1)

        contexts: tuple[Tensor, ...] | None = None
        if record_contexts:
            per_member_contexts = [out.contexts for out in flat_outputs if out.contexts is not None]
            if per_member_contexts:
                num_stages = len(per_member_contexts[0])
                contexts = tuple(_average_tensors([m[idx] for m in per_member_contexts]) for idx in range(num_stages))

        return PangolinModelOutput(
            last_hidden_state=last_hidden_state,
            probabilities=probabilities,
            contexts=contexts if output_contexts else None,
            hidden_states=contexts if output_hidden_states else None,
        )

    @property
    def output_channels(self) -> list[str]:
        channels = []
        for tissue in self.config.tissue_names:
            channels.extend(
                [
                    f"{tissue}_no_splice",
                    f"{tissue}_splice_site",
                    f"{tissue}_usage",
                ]
            )
        return channels

    def postprocess(self, outputs: PangolinModelOutput | ModelOutput | Tensor) -> tuple[Tensor, list[str]]:
        r"""
        Return Pangolin splice-site scores with semantic tissue channel names.

        Pangolin's outputs are already probability-like from the original head: two softmax splice-site channels and
        one sigmoid usage channel for each tissue. This method attaches the model-defined tissue channel names so
        direct model users and pipelines share the same output semantics.

        Args:
            outputs: The output of [`PangolinModel`][multimolecule.models.PangolinModel], or its `probabilities`
                tensor.

        Returns:
            A tuple of `(scores, channels)`, where `scores` has shape `(batch_size, sequence_length, num_tissues * 3)`
            and `channels` follows `config.tissue_names`.
        """
        probabilities = outputs if isinstance(outputs, Tensor) else outputs["probabilities"]
        return probabilities, self.output_channels


class PangolinForTokenPrediction(PangolinPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import PangolinConfig, PangolinForTokenPrediction, PangolinStageConfig
        >>> stage = PangolinStageConfig(num_blocks=1, kernel_size=3, dilation=1)
        >>> config = PangolinConfig(context=4, num_tissues=1, num_ensemble=1, num_labels=1, stages=[stage])
        >>> model = PangolinForTokenPrediction(config)
        >>> input_ids = torch.randint(5, (1, 5))
        >>> output = model(input_ids, labels=torch.rand(1, 5, 1))
        >>> output["logits"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MseLossBackward0>)
    """

    def __init__(self, config: PangolinConfig):
        super().__init__(config)
        self.model = PangolinModel(config)
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
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | PangolinTokenPredictorOutput:
        head_attention_mask = attention_mask
        if input_ids is None and inputs_embeds is not None and head_attention_mask is None:
            if isinstance(inputs_embeds, NestedTensor):
                head_attention_mask = inputs_embeds.mask
            else:
                head_attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.int, device=inputs_embeds.device)

        outputs = self.model(
            input_ids,
            attention_mask=head_attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.token_head(outputs, head_attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return PangolinTokenPredictorOutput(
            loss=loss,
            logits=logits,
            contexts=outputs.contexts,
            hidden_states=outputs.hidden_states,
        )


class PangolinModule(nn.Module):
    """A single trained Pangolin network (one ensemble member)."""

    def __init__(self, config: PangolinConfig):
        super().__init__()
        self.projection = nn.Conv1d(config.vocab_size, config.hidden_size, 1)
        self.encoder = PangolinEncoder(config)
        self.prediction = PangolinPredictionHead(config)

    def forward(
        self,
        inputs_embeds: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PangolinModuleOutput:
        embedding = self.projection(inputs_embeds)
        outputs = self.encoder(embedding, **kwargs)
        context = outputs.last_context
        probabilities = self.prediction(context).transpose(1, 2)

        return PangolinModuleOutput(
            probabilities=probabilities,
            last_hidden_state=context.transpose(1, 2),
            last_context=context,
            contexts=outputs.contexts,
        )


class PangolinEmbedding(nn.Module):
    def __init__(self, config: PangolinConfig):
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


class PangolinEncoder(nn.Module):
    def __init__(self, config: PangolinConfig):
        super().__init__()
        self.config = config
        self.context = config.context
        self.padding = config.context // 2
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, 1, padding="same")
        self.stages = nn.ModuleList([PangolinStage(config, **s) for s in config.stages])
        self.context_recorder = PangolinContextRecorder()
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PangolinModuleOutput:
        record_contexts = kwargs.get("output_contexts", self.config.output_contexts) or kwargs.get(
            "output_hidden_states", self.config.output_hidden_states
        )
        all_contexts: tuple[Tensor, ...] | None = () if record_contexts else None
        skip = self.conv(hidden_state)
        if record_contexts:
            trimmed_context = skip[:, :, self.padding : -self.padding].transpose(1, 2)
            all_contexts = all_contexts + (self.context_recorder(trimmed_context),)  # type: ignore[operator]
        for stage in self.stages:
            if self.gradient_checkpointing and self.training:
                hidden_state, skip = self._gradient_checkpointing_func(
                    stage.__call__,
                    hidden_state,
                    skip,
                    context_fn=lambda stage=stage: (nullcontext(), preserve_batch_norm_stats(stage)),
                )
            else:
                hidden_state, skip = stage(hidden_state, skip)
            if record_contexts:
                trimmed_context = skip[:, :, self.padding : -self.padding].transpose(1, 2)
                all_contexts = all_contexts + (self.context_recorder(trimmed_context),)  # type: ignore[operator]

        skip = skip[:, :, self.padding : -self.padding]
        return PangolinModuleOutput(last_context=skip, contexts=all_contexts)


class PangolinStage(nn.Module):
    def __init__(self, config: PangolinConfig, num_blocks: int = 4, kernel_size: int = 11, dilation: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PangolinBlock(config, kernel_size=kernel_size, dilation=dilation) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding="same")

    def forward(self, hidden_state: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            hidden_state = block(hidden_state)
        skip = skip + self.conv(hidden_state)
        return hidden_state, skip


class PangolinBlock(nn.Module):
    def __init__(self, config: PangolinConfig, kernel_size: int, dilation: int = 1):
        super().__init__()
        if isinstance(config.hidden_act, str):
            act1 = ACT2FN[config.hidden_act]
            act2 = ACT2FN[config.hidden_act]
        else:
            act1 = config.hidden_act
            act2 = config.hidden_act
        self.norm1 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = act1
        self.conv1 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.norm2 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act2 = act2
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


class PangolinPredictionHead(nn.Module):
    """Original Pangolin output heads.

    For each tissue the network predicts a 2-channel splice-site score (softmax over the 2 channels) and a 1-channel
    splice-site usage score (sigmoid). The concatenation of all tissue heads reproduces the upstream 12-channel output
    used to compute splice-variant-effect deltas. These heads are checkpoint state; the downstream task head is the
    shared [`TokenPredictionHead`].
    """

    def __init__(self, config: PangolinConfig):
        super().__init__()
        self.num_tissues = config.num_tissues
        self.score = nn.ModuleList([nn.Conv1d(config.hidden_size, 2, 1) for _ in range(config.num_tissues)])
        self.usage = nn.ModuleList([nn.Conv1d(config.hidden_size, 1, 1) for _ in range(config.num_tissues)])

    def forward(self, context: Tensor) -> Tensor:
        outputs = []
        for tissue in range(self.num_tissues):
            outputs.append(F.softmax(self.score[tissue](context), dim=1))
            outputs.append(torch.sigmoid(self.usage[tissue](context)))
        return torch.cat(outputs, dim=1)


class PangolinContextRecorder(nn.Module):
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
class PangolinModelOutput(ModelOutput):
    """
    Base class for outputs of the Pangolin model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Per-position encoder representation, averaged across ensemble members. Consumed by
            [`TokenPredictionHead`].
        probabilities (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_tissues * 3)`):
            Original Pangolin per-tissue splice-site score (softmax, 2 channels) and splice-site usage score (sigmoid,
            1 channel) outputs, averaged across ensemble members. These are post-activation probabilities; Pangolin has
            no pre-softmax logit surface.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True` is passed or when
            `config.output_contexts=True`):
            Tuple of `torch.FloatTensor` (one per stage of the encoder) of shape `(batch_size, sequence_length,
            hidden_size)`. Skip vectors recorded after each stage, averaged across ensemble members.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Same content as `contexts`; provided for compatibility with the Transformers hidden-states convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    probabilities: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class PangolinModuleOutput(ModelOutput):
    """Internal output of a single Pangolin ensemble member, holding post-activation per-tissue probabilities."""

    probabilities: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    last_context: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class PangolinTokenPredictorOutput(ModelOutput):
    """
    Base class for outputs of Pangolin token prediction models.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Token prediction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            Per-nucleotide prediction outputs.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True`):
            Per-stage context representations.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Per-stage context representations.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


PangolinPreTrainedModel._can_record_outputs = {
    "contexts": PangolinContextRecorder,
}
