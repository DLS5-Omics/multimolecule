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

from collections.abc import Callable
from typing import Any
from warnings import warn

import torch
import torch.nn.functional as F
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
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    OutputRecorder,
    PreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import (
    ContactPredictionHead,
    HeadOutput,
    RotaryEmbedding,
    SequencePredictionHead,
    TokenPredictionHead,
    eager_attention_forward,
)

from ..modeling_outputs import (
    ContactPredictorOutput,
    SequencePredictorOutput,
    TokenPredictorOutput,
)
from .configuration_ablang2 import AbLang2Config


class AbLang2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AbLang2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["AbLang2Layer"]

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


class AbLang2Model(AbLang2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AbLang2Config, AbLang2Model, ProteinTokenizer
        >>> config = AbLang2Config()
        >>> model = AbLang2Model(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("EVQLVESGGGLVQPGGSLRLSCAAS", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 27, 480])
        >>> output["pooler_output"].shape
        torch.Size([1, 480])
    """

    def __init__(self, config: AbLang2Config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = AbLang2Embeddings(config)
        self.encoder = AbLang2Encoder(config)
        self.pooler = AbLang2Pooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if attention_mask is None and input_ids is not None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        embedding_output = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        attn_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(embedding_output, attention_mask=attn_mask, **kwargs)
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AbLang2ForSequencePrediction(AbLang2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AbLang2Config, AbLang2ForSequencePrediction, ProteinTokenizer
        >>> config = AbLang2Config()
        >>> model = AbLang2ForSequencePrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("EVQLVESGGGLVQPGGSLRLSCAAS", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: AbLang2Config):
        super().__init__(config)
        self.model = AbLang2Model(config)
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


class AbLang2ForTokenPrediction(AbLang2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AbLang2Config, AbLang2ForTokenPrediction, ProteinTokenizer
        >>> config = AbLang2Config()
        >>> model = AbLang2ForTokenPrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("EVQLVESGGGLVQPGGSLRLSCAAS", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 25)))
        >>> output["logits"].shape
        torch.Size([1, 25, 1])
    """

    def __init__(self, config: AbLang2Config):
        super().__init__(config)
        self.model = AbLang2Model(config, add_pooling_layer=False)
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


class AbLang2ForContactPrediction(AbLang2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AbLang2Config, AbLang2ForContactPrediction, ProteinTokenizer
        >>> config = AbLang2Config()
        >>> model = AbLang2ForContactPrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("EVQLVESGGGLVQPGGSLRLSCAAS", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 25, 25)))
        >>> output["logits"].shape
        torch.Size([1, 25, 25, 1])
    """

    def __init__(self, config: AbLang2Config):
        super().__init__(config)
        self.model = AbLang2Model(config, add_pooling_layer=False)
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config
        self.require_attentions = self.contact_head.require_attentions

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
    ) -> tuple[Tensor, ...] | ContactPredictorOutput:
        if self.require_attentions:
            output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            kwargs["output_attentions"] = True
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.contact_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return ContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AbLang2ForMaskedLM(AbLang2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AbLang2Config, AbLang2ForMaskedLM, ProteinTokenizer
        >>> config = AbLang2Config()
        >>> model = AbLang2ForMaskedLM(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("EVQLVESGGGLVQPGGSLRLSCAAS", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 27, 37])
    """

    _tied_weights_keys = {
        "lm_head.decoder.bias": "lm_head.bias",
        "lm_head.decoder.weight": "model.embeddings.word_embeddings.weight",
    }

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        tied_weights = super().get_expanded_tied_weights_keys(all_submodels=all_submodels)
        if all_submodels:
            return tied_weights
        return tied_weights | self._tied_weights_keys

    def __init__(self, config: AbLang2Config):
        super().__init__(config)
        self.model = AbLang2Model(config, add_pooling_layer=False)
        self.lm_head = AbLang2LMHead(config, weight=self.model.embeddings.word_embeddings.weight)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.word_embeddings = value
        self.lm_head.decoder.weight = value.weight

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

        output = self.lm_head(outputs.last_hidden_state, labels)
        logits, loss = output.logits, output.loss

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AbLang2ForPreTraining(AbLang2ForMaskedLM):
    pass


class AbLang2Embeddings(nn.Module):
    def __init__(self, config: AbLang2Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return inputs_embeds


class AbLang2Encoder(nn.Module):
    def __init__(self, config: AbLang2Config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AbLang2Layer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        all_attentions: list[Tensor] | None = [] if output_attentions else None

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            if all_attentions is not None and attention_probs is not None:
                all_attentions.append(attention_probs)

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
            past_key_values=None,
        )


class AbLang2Layer(GradientCheckpointingLayer):
    def __init__(self, config: AbLang2Config, layer_idx: int | None = None):
        super().__init__()
        self.attention = AbLang2Attention(config, layer_idx=layer_idx)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = AbLang2Intermediate(config)
        self.output = AbLang2Output(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        residual = attention_output
        hidden_states = self.layer_norm(attention_output)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, residual)
        return hidden_states, attention_probs


class AbLang2Attention(nn.Module):
    def __init__(self, config: AbLang2Config, layer_idx: int | None = None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self = AbLang2SelfAttention(config, layer_idx=layer_idx)
        self.output = AbLang2SelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attention_output, attention_probs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        attention_output = self.output(attention_output, residual)
        return attention_output, attention_probs


class AbLang2SelfAttention(nn.Module):
    """Multi-headed self-attention with upstream-compatible adjacent-pair rotary embeddings."""

    def __init__(self, config: AbLang2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        # Upstream scales the query projection and then divides attention scores
        # by sqrt(head_dim), making the effective attention-score scale head_dim^-1.
        self.scaling = self.head_dim**-1
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.rotary_embeddings = RotaryEmbedding(
            embedding_dim=self.head_dim,
            base=config.rotary_base,
            interleaved=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = self.rotary_embeddings(query_states, key_states)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, self.all_head_size).contiguous()
        return attn_output, attn_weights


class AbLang2SelfOutput(nn.Module):
    def __init__(self, config: AbLang2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class AbLang2Intermediate(nn.Module):
    def __init__(self, config: AbLang2Config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            _activation_projection_size(config),
            bias=config.feedforward_bias,
        )
        self.activation = AbLang2SwiGLU() if config.hidden_act == "swiglu" else ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        return self.activation(hidden_states)


class AbLang2Output(nn.Module):
    def __init__(self, config: AbLang2Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.feedforward_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class AbLang2Pooler(nn.Module):
    def __init__(self, config: AbLang2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return self.activation(pooled_output)


class AbLang2LMHead(nn.Module):
    def __init__(self, config: AbLang2Config, weight: nn.Parameter | None = None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, _lm_head_projection_size(config), bias=True)
        self.activation = AbLang2SwiGLU() if config.hidden_act == "swiglu" else ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if weight is not None:
            self.decoder.weight = weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor, labels: Tensor | None = None) -> HeadOutput:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = F.linear(hidden_states, self.decoder.weight, self.bias)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return HeadOutput(logits, loss)
        return HeadOutput(logits)

    def _tie_weights(self):
        self.decoder.bias = self.bias


class AbLang2SwiGLU(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return F.silu(gate) * hidden_states


def _activation_projection_size(config: AbLang2Config) -> int:
    return config.intermediate_size * 2 if config.hidden_act == "swiglu" else config.intermediate_size


def _lm_head_projection_size(config: AbLang2Config) -> int:
    return config.hidden_size * 2 if config.hidden_act == "swiglu" else config.hidden_size


AbLang2PreTrainedModel._can_record_outputs = {
    "hidden_states": AbLang2Layer,
    "attentions": OutputRecorder(AbLang2Attention, index=1, layer_name="attention"),
}
