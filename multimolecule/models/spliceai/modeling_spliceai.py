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
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_spliceai import SpliceAiConfig


class SpliceAiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpliceAiConfig
    base_model_prefix = "spliceai"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SpliceAiBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


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
        self.embeddings = SpliceAiEmbedding(config)
        self.networks = nn.ModuleList([SpliceAiModule(config) for _ in range(5)])

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        output_contexts: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SpliceAiModelOutput | Tuple[Tensor, Tuple[Tensor, ...]] | Tensor:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        output_contexts = output_contexts if output_contexts is not None else self.config.output_contexts
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        all_outputs = [
            module(
                embedding_output,
                output_contexts=output_contexts,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            for module in self.networks
        ]

        if not return_dict:
            return tuple(average_output(output) for output in zip(*all_outputs))

        outputs: Dict = {k: [outputs[k] for outputs in all_outputs] for k in all_outputs[0]}
        for key, output in outputs.items():
            outputs[key] = average_output(output)

        return SpliceAiModelOutput(**outputs)


class SpliceAiEmbedding(nn.Module):
    def __init__(self, config: SpliceAiConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_tokens = self.vocab_size + 1
        self.context = config.context
        self.padding = config.context // 2

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = F.one_hot(input_ids, num_classes=self.num_tokens)[..., : self.vocab_size].float()
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        batch_size = inputs_embeds.size(0)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        padding = torch.zeros(batch_size, self.vocab_size, self.padding, device=inputs_embeds.device)
        inputs_embeds = torch.cat([padding, inputs_embeds, padding], dim=2)
        return inputs_embeds


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
        output_contexts: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SpliceAiModelOutput | Tuple[Tensor, Tuple[Tensor, ...]] | Tensor:
        embedding = self.projection(inputs_embeds)
        outputs = self.encoder(
            embedding,
            output_contexts=output_contexts,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        context = outputs.last_context if return_dict else outputs[0]
        logits = self.prediction(context).transpose(1, 2)

        loss = self.criterion(logits, labels) if labels is not None else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SpliceAiModelOutput(
            loss=loss, logits=logits, contexts=outputs.contexts, hidden_states=outputs.hidden_states
        )


class SpliceAiEncoder(nn.Module):

    def __init__(self, config: SpliceAiConfig):
        super().__init__()
        self.context = config.context
        self.padding = config.context // 2
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, 1, padding="same")
        self.stages = nn.ModuleList([SpliceAiStage(config, **s) for s in config.stages])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        output_contexts: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SpliceAiModuleOutput | Tuple[Tensor, Tuple[Tensor, ...]] | Tensor:
        contexts = () if output_contexts else None
        hidden_states = () if output_hidden_states else None

        context = self.conv(hidden_state)
        for stage in self.stages:
            if output_contexts:
                contexts = contexts + (context[:, :, self.padding : -self.padding].transpose(1, 2),)  # type: ignore[operator] # noqa: E501
            if output_hidden_states:
                hidden_states = hidden_states + (context[:, :, self.padding : -self.padding].transpose(1, 2),)  # type: ignore[operator] # noqa: E501

            if self.gradient_checkpointing and self.training:
                hidden_state, context = self._gradient_checkpointing_func(stage.__call__, hidden_state, context)
            else:
                hidden_state, context = stage(hidden_state, context)

        context = context[:, :, self.padding : -self.padding]

        if output_contexts:
            contexts = contexts + (context.transpose(1, 2),)  # type: ignore[operator]
        if output_hidden_states:
            hidden_states = hidden_states + (context.transpose(1, 2),)  # type: ignore[operator]

        if not return_dict:
            return context, hidden_states if output_hidden_states else context

        return SpliceAiModuleOutput(last_context=context, contexts=contexts, hidden_states=hidden_states)


class SpliceAiStage(nn.Module):
    def __init__(self, config: SpliceAiConfig, num_blocks: int = 1, kernel_size=11, dilation: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[SpliceAiBlock(config, kernel_size=kernel_size, dilation=dilation) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            padding="same",
        )

    def forward(self, hidden_state: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        hidden_state = self.blocks(hidden_state)
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


def average_output(output: Tuple[Tensor, ...] | Tuple[Tuple[Tensor, ...], ...]) -> Tensor | Tuple[Tensor, ...]:
    if isinstance(output[0], Tensor):
        return torch.mean(torch.stack(output), dim=0)
    return tuple(torch.mean(torch.stack(o), dim=0) for o in zip(*output))


@dataclass
class SpliceAiModelOutput(ModelOutput):
    """
    Base class for outputs of SpliceAI model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: Tuple[torch.FloatTensor, ...] | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class SpliceAiModuleOutput(ModelOutput):
    last_context: torch.FloatTensor
    contexts: Tuple[torch.FloatTensor, ...] | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
