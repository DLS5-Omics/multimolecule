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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import can_return_tuple

from multimolecule.modules import TokenPredictionHead, preserve_batch_norm_stats

from .configuration_openspliceai import OpenSpliceAiConfig


class OpenSpliceAiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OpenSpliceAiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenSpliceAiBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class OpenSpliceAiModel(OpenSpliceAiPreTrainedModel):
    """
    The bare OpenSpliceAI backbone producing per-nucleotide context representations.

    Examples:
        >>> import torch
        >>> from multimolecule import OpenSpliceAiConfig, OpenSpliceAiModel
        >>> config = OpenSpliceAiConfig()
        >>> model = OpenSpliceAiModel(config)
        >>> input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 5, 32])
    """

    def __init__(self, config: OpenSpliceAiConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = OpenSpliceAiEmbedding(config)
        self.encoder = OpenSpliceAiEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        output_contexts: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> OpenSpliceAiModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        output_contexts = output_contexts if output_contexts is not None else self.config.output_contexts
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        record_contexts = bool(output_contexts) or bool(output_hidden_states)

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

        encoder_outputs = self.encoder(embedding_output, output_hidden_states=record_contexts)
        last_hidden_state = encoder_outputs.last_hidden_state.transpose(1, 2)
        contexts = None
        if encoder_outputs.hidden_states is not None:
            contexts = tuple(hidden_state.transpose(1, 2) for hidden_state in encoder_outputs.hidden_states)

        return OpenSpliceAiModelOutput(
            last_hidden_state=last_hidden_state,
            contexts=contexts if output_contexts else None,
            hidden_states=contexts if output_hidden_states else None,
        )


class OpenSpliceAiForTokenPrediction(OpenSpliceAiPreTrainedModel):
    """
    OpenSpliceAI model for per-nucleotide splice-site classification (neither / acceptor / donor).

    Examples:
        >>> import torch
        >>> from multimolecule import OpenSpliceAiConfig, OpenSpliceAiForTokenPrediction
        >>> config = OpenSpliceAiConfig()
        >>> model = OpenSpliceAiForTokenPrediction(config)
        >>> input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        >>> output = model(input_ids, labels=torch.randint(3, (1, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 3])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    def __init__(self, config: OpenSpliceAiConfig):
        super().__init__(config)
        self.model = OpenSpliceAiModel(config)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_labels == 3:
            return ["no_splice", "acceptor", "donor"]
        return [f"label_{index}" for index in range(self.config.num_labels)]

    def postprocess(self, outputs: OpenSpliceAiTokenPredictorOutput | ModelOutput | Tensor) -> tuple[Tensor, list[str]]:
        r"""
        Return OpenSpliceAI splice-site probabilities with semantic channel names.

        Args:
            outputs: The output of
                [`OpenSpliceAiForTokenPrediction`][multimolecule.models.OpenSpliceAiForTokenPrediction], or its
                `logits` tensor.

        Returns:
            A tuple of `(scores, channels)`, where `scores` is softmax-normalized over splice-site classes.
        """
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]
        return logits.softmax(dim=-1), self.output_channels

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_contexts: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> OpenSpliceAiTokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_contexts=output_contexts,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return OpenSpliceAiTokenPredictorOutput(
            loss=loss,
            logits=logits,
            contexts=outputs.contexts,
            hidden_states=outputs.hidden_states,
        )


class OpenSpliceAiEmbedding(nn.Module):
    """One-hot encodes nucleotide tokens and zero-pads the context window on each side."""

    def __init__(self, config: OpenSpliceAiConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_tokens = self.vocab_size + 1
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
            inputs_embeds = F.one_hot(input_ids, num_classes=self.num_tokens)[..., : self.vocab_size].to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        if self.padding > 0:
            batch_size = inputs_embeds.size(0)
            pad = torch.zeros(
                batch_size, self.vocab_size, self.padding, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = torch.cat([pad, inputs_embeds, pad], dim=2)
        return inputs_embeds


class OpenSpliceAiEncoder(nn.Module):
    def __init__(self, config: OpenSpliceAiConfig):
        super().__init__()
        self.config = config
        self.padding = config.context // 2
        self.projection = nn.Conv1d(config.vocab_size, config.hidden_size, 1)
        self.skip = nn.Conv1d(config.hidden_size, config.hidden_size, 1)
        self.stages = nn.ModuleList([OpenSpliceAiStage(config, **s) for s in config.stages])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: Tensor,
        output_hidden_states: bool = False,
    ) -> OpenSpliceAiEncoderOutput:
        hidden_state = self.projection(inputs_embeds)
        context = self.skip(hidden_state)
        all_hidden_states: Tuple[Tensor, ...] | None = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (self._crop(context),)  # type: ignore[operator]
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
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (self._crop(context),)  # type: ignore[operator]

        context = self._crop(context)
        return OpenSpliceAiEncoderOutput(last_hidden_state=context, hidden_states=all_hidden_states)

    def _crop(self, hidden_state: Tensor) -> Tensor:
        if self.padding > 0:
            return hidden_state[:, :, self.padding : -self.padding]
        return hidden_state


class OpenSpliceAiStage(nn.Module):
    def __init__(self, config: OpenSpliceAiConfig, num_blocks: int = 4, kernel_size: int = 11, dilation: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [OpenSpliceAiBlock(config, kernel_size=kernel_size, dilation=dilation) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, 1)

    def forward(self, hidden_state: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        for block in self.blocks:
            hidden_state = block(hidden_state)
        context = self.conv(hidden_state) + context
        return hidden_state, context


class OpenSpliceAiBlock(nn.Module):
    """A pre-activation dilated residual unit (BN -> activation -> Conv) x 2."""

    def __init__(self, config: OpenSpliceAiConfig, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        hidden_act_kwargs = getattr(config, "hidden_act_kwargs", {})
        if isinstance(config.hidden_act, str) and hidden_act_kwargs:
            act1 = type(ACT2FN[config.hidden_act])(**hidden_act_kwargs)
            act2 = type(ACT2FN[config.hidden_act])(**hidden_act_kwargs)
        elif isinstance(config.hidden_act, str):
            act1 = ACT2FN[config.hidden_act]
            act2 = ACT2FN[config.hidden_act]
        else:
            act1 = config.hidden_act
            act2 = config.hidden_act
        self.norm1 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = act1
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, dilation=dilation, padding=padding)
        self.norm2 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act2 = act2
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, dilation=dilation, padding=padding)

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.conv1(self.act1(self.norm1(hidden_state)))
        hidden_state = self.conv2(self.act2(self.norm2(hidden_state)))
        return hidden_state + residual


@dataclass
class OpenSpliceAiModelOutput(ModelOutput):
    """
    Base class for outputs of the OpenSpliceAI backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Per-nucleotide context representation after the dilated residual stack.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True`):
            Per-stage context representations.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Per-stage context representations.
    """

    last_hidden_state: torch.FloatTensor | None = None
    contexts: Tuple[torch.FloatTensor, ...] | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class OpenSpliceAiTokenPredictorOutput(ModelOutput):
    """
    Base class for outputs of OpenSpliceAI token prediction models.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Token prediction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            Per-nucleotide splice-site classification scores.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True`):
            Per-stage context representations.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Per-stage context representations.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: Tuple[torch.FloatTensor, ...] | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class OpenSpliceAiEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
