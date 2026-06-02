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

from inspect import signature
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import RotaryEmbedding, SequencePredictionHead, TokenPredictionHead, attention_forward

from ..modeling_outputs import SequencePredictorOutput, TokenPredictorOutput
from .configuration_progen2 import ProGen2Config

_CREATE_CAUSAL_MASK_PARAMETERS = set(signature(create_causal_mask).parameters)


class ProGen2PreTrainedModel(PreTrainedModel):
    config_class = ProGen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["ProGen2Block"]
    _skip_keys_device_placement = ["past_key_values"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class ProGen2Model(ProGen2PreTrainedModel):
    """
    Note:
        When gradient checkpointing is enabled (``model.gradient_checkpointing_enable()``), ``use_cache`` is
        incompatible with recomputation and should be set to ``False``; past key-value caching will not function
        correctly under gradient checkpointing.

    Examples:
        >>> import torch
        >>> from multimolecule import ProGen2Config, ProGen2Model
        >>> config = ProGen2Config()
        >>> model = ProGen2Model(config)
    """

    def __init__(self, config: ProGen2Config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = ProGen2Embeddings(config)
        self.decoder = ProGen2Decoder(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = min(config.rotary_dim, head_dim) if config.rotary_dim is not None else head_dim
        # ProGen2/GPT-J use the interleaved (rotate_every_two) RoPE variant; the shared RotaryEmbedding
        # covers it. cos/sin are computed at the model level from explicit position_ids (KV-cache /
        # generation / packed-NestedTensor per-sequence positions) and applied inside attention.
        self.rotary_emb = RotaryEmbedding(embedding_dim=rotary_dim, interleaved=True)
        self.pooler = ProGen2Pooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        is_nested = isinstance(input_ids, NestedTensor) or isinstance(inputs_embeds, NestedTensor)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        else:
            inputs_embeds = self.embeddings(inputs_embeds=inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            if is_nested:
                reference = input_ids if input_ids is not None else inputs_embeds[..., 0]
                position_ids = torch.cumsum(torch.ones_like(reference, dtype=torch.long), dim=1) - 1
            else:
                position_ids = cache_position.unsqueeze(0)

        if is_nested:
            causal_mask = None
        else:
            causal_mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if "cache_position" in _CREATE_CAUSAL_MASK_PARAMETERS:
                causal_mask_kwargs["cache_position"] = cache_position
            causal_mask = create_causal_mask(**causal_mask_kwargs)

        position_embeddings = self.rotary_emb.cos_sin_tables(position_ids, dtype=inputs_embeds.dtype)

        decoder_outputs = self.decoder(
            inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = self.layer_norm(decoder_outputs.last_hidden_state)
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            past_key_values=decoder_outputs.past_key_values,
        )


class ProGen2ForCausalLM(ProGen2PreTrainedModel, GenerationMixin):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProGen2Config, ProGen2ForCausalLM
        >>> config = ProGen2Config()
        >>> model = ProGen2ForCausalLM(config)
    """

    def __init__(self, config: ProGen2Config):
        super().__init__(config)
        self.model = ProGen2Model(config, add_pooling_layer=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits; the loss path casts to float32 for cross-entropy stability
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=lm_logits.to(torch.float32),
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ProGen2ForSequencePrediction(ProGen2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProGen2Config, ProGen2ForSequencePrediction
        >>> config = ProGen2Config()
        >>> model = ProGen2ForSequencePrediction(config)
    """

    def __init__(self, config: ProGen2Config):
        super().__init__(config)
        self.model = ProGen2Model(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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


class ProGen2ForTokenPrediction(ProGen2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProGen2Config, ProGen2ForTokenPrediction
        >>> config = ProGen2Config()
        >>> model = ProGen2ForTokenPrediction(config)
    """

    def __init__(self, config: ProGen2Config):
        super().__init__(config)
        self.model = ProGen2Model(config, add_pooling_layer=False)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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


class ProGen2ForPreTraining(ProGen2ForCausalLM):
    pass


class ProGen2Embeddings(nn.Module):
    def __init__(self, config: ProGen2Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: Tensor | None = None, inputs_embeds: Tensor | None = None) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return self.dropout(inputs_embeds)


class ProGen2Decoder(nn.Module):
    def __init__(self, config: ProGen2Config):
        super().__init__()
        self.layers = nn.ModuleList([ProGen2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class ProGen2Block(GradientCheckpointingLayer):
    """GPT-J-style transformer block with parallel attention and MLP.

    Both attention and MLP receive the same layer-normed input, and their
    outputs are summed with the residual in a single addition.
    """

    def __init__(self, config: ProGen2Config, layer_idx: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ProGen2Attention(config, layer_idx=layer_idx)
        self.mlp = ProGen2MLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        # Parallel attention + MLP (GPT-J style)
        attn_output, _ = self.attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = residual + attn_output + feed_forward_output

        return hidden_states


class ProGen2Attention(nn.Module):
    """Multi-headed attention with partial rotary position embeddings (GPT-J style).

    Supports multiple attention backends (eager, SDPA, Flash Attention 2)
    through the shared MultiMolecule attention dispatch.
    """

    def __init__(self, config: ProGen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != config.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={config.hidden_size}, "
                f"num_attention_heads={self.num_attention_heads})"
            )
        self.scaling = self.head_dim**-0.5 if config.scale_attn_weights else 1.0
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.rotary_dim = min(config.rotary_dim, self.head_dim) if config.rotary_dim is not None else self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply partial rotary position embeddings
        cos, sin = position_embeddings
        if self.rotary_dim < self.head_dim:
            q_rot = query_states[..., : self.rotary_dim]
            q_pass = query_states[..., self.rotary_dim :]
            k_rot = key_states[..., : self.rotary_dim]
            k_pass = key_states[..., self.rotary_dim :]

            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_dim,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            attn_implementation=self.config._attn_implementation,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class ProGen2MLP(nn.Module):
    def __init__(self, config: ProGen2Config):
        super().__init__()
        self.fc_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ProGen2Pooler(nn.Module):
    def __init__(self, config: ProGen2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply ProGen2/GPT-J-style (interleaved ``rotate_every_two``) rotary embeddings to query and key.

    Delegates to the shared :meth:`RotaryEmbedding.apply_rotary` with ``interleaved=True`` so the
    rotation math lives in one place.

    Args:
        q: Query tensor of shape ``(batch, num_heads, seq_len, rotary_dim)``.
        k: Key tensor of shape ``(batch, num_heads, seq_len, rotary_dim)``.
        cos: Cosine tensor of shape ``(batch, 1, seq_len, rotary_dim)``.
        sin: Sine tensor of shape ``(batch, 1, seq_len, rotary_dim)``.
    """
    return (
        RotaryEmbedding.apply_rotary(q, cos, sin, interleaved=True),
        RotaryEmbedding.apply_rotary(k, cos, sin, interleaved=True),
    )


ProGen2PreTrainedModel._can_record_outputs = {
    "hidden_states": ProGen2Block,
    "attentions": OutputRecorder(ProGen2Attention, index=1),
}
