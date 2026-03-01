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
from collections.abc import Callable
from typing import Any, Tuple
from warnings import warn

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import (
    ContactPredictionHead,
    MaskedLMHead,
    SequencePredictionHead,
    TokenPredictionHead,
    eager_attention_forward,
)

from ..modeling_outputs import ContactPredictorOutput, SequencePredictorOutput, TokenPredictorOutput
from .configuration_dnaberts import DnaBertSConfig


class DnaBertSPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DnaBertSConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["DnaBertSLayer", "DnaBertSEmbeddings"]


class DnaBertSModel(DnaBertSPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DnaBertSConfig, DnaBertSModel
        >>> config = DnaBertSConfig()
        >>> model = DnaBertSModel(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (1, 16))
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 16, 768])
        >>> output["pooler_output"].shape
        torch.Size([1, 768])
    """

    def __init__(self, config: DnaBertSConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = DnaBertSEmbeddings(config)
        self.encoder = DnaBertSEncoder(config)
        self.pooler = DnaBertSPooler(config) if add_pooling_layer else None

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
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        r"""
        Args:
            encoder_hidden_states:
                Shape: `(batch_size, sequence_length, hidden_size)`

                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask:
                Shape: `(batch_size, sequence_length)`

                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values:
                Tuple of length `config.n_layers` with each tuple having 4 tensors of shape
                `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)

                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache:
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        if isinstance(input_ids, NestedTensor) and attention_mask is None:
            attention_mask = input_ids.mask
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            device = input_ids.device
            seq_length = input_ids.shape[1]
        else:
            device = inputs_embeds.device  # type: ignore[union-attr]
            seq_length = inputs_embeds.shape[1]  # type: ignore[union-attr]

        # past_key_values_length
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

        if attention_mask is None and input_ids is not None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        attention_mask, encoder_attention_mask = self._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        cache_position,
        past_key_values,
    ):
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config, input_embeds=embedding_output, attention_mask=attention_mask
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask


class DnaBertSForSequencePrediction(DnaBertSPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DnaBertSConfig, DnaBertSForSequencePrediction
        >>> config = DnaBertSConfig()
        >>> model = DnaBertSForSequencePrediction(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (1, 16))
        >>> output = model(input_ids, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: DnaBertSConfig):
        super().__init__(config)
        self.model = DnaBertSModel(config)
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
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
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


class DnaBertSForTokenPrediction(DnaBertSPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DnaBertSConfig, DnaBertSForTokenPrediction
        >>> config = DnaBertSConfig()
        >>> model = DnaBertSForTokenPrediction(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (1, 16))
        >>> output = model(input_ids, labels=torch.randint(2, (1, 14)))
        >>> output["logits"].shape
        torch.Size([1, 14, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: DnaBertSConfig):
        super().__init__(config)
        self.model = DnaBertSModel(config, add_pooling_layer=False)
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
    ) -> Tuple[Tensor, ...] | TokenPredictorOutput:
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


class DnaBertSForContactPrediction(DnaBertSPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DnaBertSConfig, DnaBertSForContactPrediction
        >>> config = DnaBertSConfig()
        >>> model = DnaBertSForContactPrediction(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (1, 16))
        >>> output = model(input_ids, labels=torch.randint(2, (1, 14, 14)))
        >>> output["logits"].shape
        torch.Size([1, 14, 14, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: DnaBertSConfig):
        super().__init__(config)
        self.model = DnaBertSModel(config, add_pooling_layer=False)
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
    ) -> Tuple[Tensor, ...] | ContactPredictorOutput:
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


class DnaBertSForMaskedLM(DnaBertSPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DnaBertSConfig, DnaBertSForMaskedLM
        >>> config = DnaBertSConfig()
        >>> model = DnaBertSForMaskedLM(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (1, 16))
        >>> output = model(input_ids, labels=input_ids)
        >>> output["logits"].shape
        torch.Size([1, 16, 4096])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    _tied_weights_keys = {
        "lm_head.decoder.weight": "model.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config: DnaBertSConfig):
        super().__init__(config)
        if config.is_decoder:
            warn(
                "If you want to use `DnaBertSForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.model = DnaBertSModel(config, add_pooling_layer=False)
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
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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


class DnaBertSForPreTraining(DnaBertSForMaskedLM):
    pass


class DnaBertSEmbeddings(nn.Module):
    """
    Construct the embeddings from word and token_type embeddings.
    DNABERT-S uses ALiBi for position encoding, so no position embeddings are needed.
    """

    def __init__(self, config: DnaBertSConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]  # type: ignore[union-attr]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # DNABERT-S does not use token_type_ids
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DnaBertSEncoder(nn.Module):
    def __init__(self, config: DnaBertSConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.layer = nn.ModuleList([DnaBertSLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        # Pre-build ALiBi tensor at starting size
        self._current_alibi_size = config.alibi_starting_size
        self.alibi = self._build_alibi_tensor(config.alibi_starting_size, torch.device("cpu"), torch.float32)

    def _build_alibi_tensor(self, size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> list[float]:
            def get_slopes_power_of_2(n_heads: int) -> list[float]:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.tensor(_get_alibi_head_slopes(n_heads), device=device, dtype=dtype)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        return alibi.unsqueeze(0)  # [1, n_heads, size, size]

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
        seq_len = hidden_states.shape[1]

        # Compute ALiBi bias for the current sequence length
        if seq_len > self._current_alibi_size:
            self.alibi = self._build_alibi_tensor(seq_len, hidden_states.device, hidden_states.dtype)
            self._current_alibi_size = seq_len
        alibi = self.alibi[:, :, :seq_len, :seq_len].to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Add ALiBi bias to attention mask
        if attention_mask is not None:
            attention_mask = attention_mask + alibi
        else:
            attention_mask = alibi

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DnaBertSLayer(GradientCheckpointingLayer):
    def __init__(self, config: DnaBertSConfig, layer_idx: int | None = None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DnaBertSAttention(config, layer_idx=layer_idx, is_causal=config.is_decoder)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = DnaBertSAttention(
                config,
                layer_idx=layer_idx,
                is_causal=False,
                is_cross_attention=True,
            )
        self.mlp = DnaBertSGatedMlp(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            cross_attention_output, _ = self.crossattention(
                attention_output,
                None,  # attention_mask
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        return self.mlp(attention_output)


class DnaBertSGatedMlp(nn.Module):
    def __init__(self, config: DnaBertSConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.up_proj(hidden_states)
        gated = hidden_states[..., : self.up_proj.out_features // 2]
        non_gated = hidden_states[..., self.up_proj.out_features // 2 :]
        hidden_states = self.activation(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class DnaBertSAttention(nn.Module):
    def __init__(
        self,
        config: DnaBertSConfig,
        layer_idx: int | None = None,
        is_causal: bool = False,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = DnaBertSCrossAttention if is_cross_attention else DnaBertSSelfAttention
        self.self = attention_class(
            config,
            layer_idx=layer_idx,
            is_causal=is_causal,
        )
        self.output = DnaBertSSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        if self.is_cross_attention:
            attention_mask = encoder_attention_mask
            attention_output, attn_weights = self.self(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            attention_output, attn_weights = self.self(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


class DnaBertSSelfAttention(nn.Module):
    def __init__(
        self,
        config: DnaBertSConfig,
        layer_idx: int | None = None,
        is_causal: bool | None = None,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)

        self.is_decoder = config.is_decoder
        self.is_causal = config.is_decoder if is_causal is None else is_causal
        self.layer_idx = layer_idx

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError("layer_idx must be set when using past_key_values.")

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if past_key_values is not None:
            current_past_key_values = (
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values
            )
            key_layer, value_layer = current_past_key_values.update(
                key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
            )

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_bias = attention_mask

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_bias,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(hidden_states.shape[:-1] + (self.all_head_size,)).contiguous()
        return attn_output, attn_weights


class DnaBertSCrossAttention(nn.Module):
    def __init__(
        self,
        config: DnaBertSConfig,
        layer_idx: int | None = None,
        is_causal: bool | None = None,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)

        self.is_decoder = config.is_decoder
        self.is_causal = config.is_decoder if is_causal is None else is_causal
        self.layer_idx = layer_idx

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided for cross-attention.")
        mixed_query_layer = self.query(hidden_states)
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError("layer_idx must be set when using past_key_values.")

        if past_key_values is not None:
            if not isinstance(past_key_values, EncoderDecoderCache):
                raise ValueError("Cross-attention caching requires EncoderDecoderCache.")
            is_updated = past_key_values.is_updated.get(self.layer_idx) if self.layer_idx is not None else False
            if is_updated:
                key_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
                value_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].values
            else:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
                key_layer, value_layer = past_key_values.cross_attention_cache.update(
                    key_layer, value_layer, self.layer_idx
                )
                past_key_values.is_updated[self.layer_idx] = True
        else:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_bias = attention_mask

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_bias,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(hidden_states.shape[:-1] + (self.all_head_size,)).contiguous()
        return attn_output, attn_weights


class DnaBertSSelfOutput(nn.Module):
    def __init__(self, config: DnaBertSConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler
class DnaBertSPooler(nn.Module):
    def __init__(self, config: DnaBertSConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


DnaBertSPreTrainedModel._can_record_outputs = {
    "hidden_states": DnaBertSLayer,
    "attentions": OutputRecorder(DnaBertSAttention, index=1, layer_name="attention"),
}
