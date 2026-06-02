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

from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import init
from transformers.activations import ACT2FN
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults

from multimolecule.modules import MaskedLMHead, SequencePredictionHead, TokenPredictionHead, attention_forward

from ..modeling_outputs import SequencePredictorOutput, TokenPredictorOutput
from .configuration_ablang import AbLangConfig


class AbLangPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AbLangConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["AbLangLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class AbLangModel(AbLangPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.ablang import AbLangConfig, AbLangModel
        >>> config = AbLangConfig()
        >>> model = AbLangModel(config)
        >>> input_ids = torch.tensor([[1, 9, 23, 21, 15, 2]])
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 6, 768])
        >>> output["pooler_output"].shape
        torch.Size([1, 768])
    """

    def __init__(self, config: AbLangConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = AbLangEmbeddings(config)
        self.encoder = AbLangEncoder(config)
        self.pooler = AbLangPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @merge_with_config_defaults
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (
            attention_mask is None
            and input_ids is not None
            and not isinstance(input_ids, NestedTensor)
            and self.pad_token_id is not None
        ):
            attention_mask = input_ids.ne(self.pad_token_id)
        elif attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(embedding_output, NestedTensor) and attention_mask is None:
            encoder_attention_mask = None
        else:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
            )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=encoder_attention_mask,
            output_hidden_states=kwargs.get("output_hidden_states", self.config.output_hidden_states),
            output_attentions=kwargs.get("output_attentions", self.config.output_attentions),
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = (
            self.pooler(sequence_output, attention_mask=attention_mask, input_ids=input_ids) if self.pooler else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AbLangForSequencePrediction(AbLangPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.ablang import AbLangConfig, AbLangForSequencePrediction
        >>> config = AbLangConfig()
        >>> model = AbLangForSequencePrediction(config)
        >>> input_ids = torch.tensor([[1, 9, 23, 21, 15, 2]])
        >>> output = model(input_ids, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: AbLangConfig):
        super().__init__(config)
        self.model = AbLangModel(config)
        self.num_labels = config.num_labels
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

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
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AbLangForTokenPrediction(AbLangPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.ablang import AbLangConfig, AbLangForTokenPrediction
        >>> config = AbLangConfig()
        >>> model = AbLangForTokenPrediction(config)
        >>> input_ids = torch.tensor([[1, 9, 23, 21, 15, 2]])
        >>> output = model(input_ids, labels=torch.randint(2, (1, 4)))
        >>> output["logits"].shape
        torch.Size([1, 4, 1])
    """

    def __init__(self, config: AbLangConfig):
        super().__init__(config)
        self.model = AbLangModel(config, add_pooling_layer=False)
        self.num_labels = config.num_labels
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
    ) -> tuple[Tensor, ...] | TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return TokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AbLangForMaskedLM(AbLangPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.ablang import AbLangConfig, AbLangForMaskedLM
        >>> config = AbLangConfig()
        >>> model = AbLangForMaskedLM(config)
        >>> input_ids = torch.tensor([[1, 9, 23, 21, 15, 2]])
        >>> output = model(input_ids, labels=input_ids)
        >>> output["logits"].shape
        torch.Size([1, 6, 37])
    """

    _tied_weights_keys = {
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        tied_weights = super().get_expanded_tied_weights_keys(all_submodels=all_submodels)
        if all_submodels:
            return tied_weights
        return tied_weights | self._tied_weights_keys

    def __init__(self, config: AbLangConfig):
        super().__init__(config)
        self.model = AbLangModel(config, add_pooling_layer=False)
        self.lm_head = MaskedLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings
        if hasattr(self.lm_head, "bias"):
            self.lm_head.bias = embeddings.bias

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.lm_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AbLangForPreTraining(AbLangForMaskedLM):
    pass


class AbLangEmbeddings(nn.Module):
    """Token and absolute position embeddings."""

    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=0,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
    ) -> Tensor | NestedTensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_ids = self.create_position_ids(input_ids, attention_mask, inputs_embeds)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)

    def create_position_ids(
        self,
        input_ids: Tensor | NestedTensor | None,
        attention_mask: Tensor | None,
        inputs_embeds: Tensor | NestedTensor,
    ) -> Tensor | NestedTensor:
        if input_ids is not None:
            mask = input_ids.ne(self.padding_idx)
        elif attention_mask is not None:
            mask = attention_mask.to(torch.bool)
        elif isinstance(inputs_embeds, NestedTensor):
            mask = inputs_embeds.mask
        else:
            mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        position_ids = torch.cumsum(mask.to(torch.long), dim=1) * mask.to(torch.long)
        if torch.any(position_ids >= self.max_position_embeddings):
            raise ValueError(
                f"AbLang position ids must be less than max_position_embeddings={self.max_position_embeddings}."
            )
        return position_ids


class AbLangEncoder(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.layer = nn.ModuleList([AbLangLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        all_attentions: list[Tensor] | None = [] if output_attentions else None

        for layer_module in self.layer:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states, attention_probs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )
            if all_attentions is not None and attention_probs is not None:
                all_attentions.append(attention_probs)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )


class AbLangLayer(GradientCheckpointingLayer):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.attention = AbLangAttention(config)
        self.intermediate = AbLangIntermediate(config)
        self.output = AbLangOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class AbLangAttention(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.self = AbLangSelfAttention(config)
        self.output = AbLangSelfOutput(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        self_output, attention_probs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_output, hidden_states)
        attention_output = self.layer_norm(attention_output)
        return attention_output, attention_probs


class AbLangSelfAttention(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, hidden_states: Tensor) -> Tensor:
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.attention_head_size)

        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        context_layer, attention_probs = attention_forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attn_implementation=self.config._attn_implementation,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=False,
            output_attentions=output_attentions,
        )

        context_layer = context_layer.reshape(*input_shape, self.all_head_size).contiguous()
        return context_layer, attention_probs if output_attentions else None


class AbLangSelfOutput(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class AbLangIntermediate(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        return self.intermediate_act_fn(hidden_states)


class AbLangOutput(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return self.layer_norm(hidden_states)


class AbLangPooler(nn.Module):
    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        hidden_states: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        input_ids: Tensor | NestedTensor | None = None,
    ) -> Tensor:
        if input_ids is not None:
            mask = input_ids.ne(self.pad_token_id)
            mask = mask & input_ids.ne(self.bos_token_id)
            mask = mask & input_ids.ne(self.eos_token_id)
        elif attention_mask is None:
            mask = hidden_states.new_ones(hidden_states.shape[:2], dtype=torch.bool)
        else:
            mask = attention_mask.to(torch.bool)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=hidden_states.dtype)
        return (hidden_states * mask.unsqueeze(-1).to(dtype=hidden_states.dtype)).sum(dim=1) / denominator
