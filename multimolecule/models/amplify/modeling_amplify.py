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
from transformers import initialization as init
from transformers.masking_utils import create_bidirectional_mask
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
    RotaryEmbedding,
    SequencePredictionHead,
    TokenPredictionHead,
    eager_attention_forward,
)

from ..modeling_outputs import ContactPredictorOutput, SequencePredictorOutput, TokenPredictorOutput
from .configuration_amplify import AmplifyConfig


class AmplifyPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AmplifyConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["AmplifyLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # AMPLIFY uses uniform initialization for both embeddings and linears, scaled by ``initializer_range``.
        # Falling through to the parent ``_init_weights`` would apply normal-distribution init instead.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.uniform_(module.weight, -std, std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.uniform_(module.weight, -std, std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif "RMSNorm" in module.__class__.__name__ or "LayerNorm" in module.__class__.__name__:
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)


class AmplifyModel(AmplifyPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AmplifyConfig, AmplifyModel, ProteinTokenizer
        >>> config = AmplifyConfig()
        >>> model = AmplifyModel(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 11, 640])
        >>> output["pooler_output"].shape
        torch.Size([1, 640])
    """

    def __init__(self, config: AmplifyConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = AmplifyEmbeddings(config)
        self.encoder = AmplifyEncoder(config)
        self.pooler = AmplifyPooler(config) if add_pooling_layer else None

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
        if isinstance(input_ids, NestedTensor) and attention_mask is None:
            attention_mask = input_ids.mask
        if isinstance(inputs_embeds, NestedTensor) and attention_mask is None:
            attention_mask = inputs_embeds.mask
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if attention_mask is None and input_ids is not None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        embedding_output = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        attn_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=embedding_output, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(embedding_output, attention_mask=attn_mask, **kwargs)
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class AmplifyForSequencePrediction(AmplifyPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AmplifyConfig, AmplifyForSequencePrediction, ProteinTokenizer
        >>> config = AmplifyConfig()
        >>> model = AmplifyForSequencePrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AmplifyConfig):
        super().__init__(config)
        self.model = AmplifyModel(config)
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


class AmplifyForTokenPrediction(AmplifyPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AmplifyConfig, AmplifyForTokenPrediction, ProteinTokenizer
        >>> config = AmplifyConfig()
        >>> model = AmplifyForTokenPrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 9)))
        >>> output["logits"].shape
        torch.Size([1, 9, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AmplifyConfig):
        super().__init__(config)
        self.model = AmplifyModel(config, add_pooling_layer=False)
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


class AmplifyForContactPrediction(AmplifyPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AmplifyConfig, AmplifyForContactPrediction, ProteinTokenizer
        >>> config = AmplifyConfig()
        >>> model = AmplifyForContactPrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 9, 9)))
        >>> output["logits"].shape
        torch.Size([1, 9, 9, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AmplifyConfig):
        super().__init__(config)
        self.model = AmplifyModel(config, add_pooling_layer=False)
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


class AmplifyForMaskedLM(AmplifyPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AmplifyConfig, AmplifyForMaskedLM, ProteinTokenizer
        >>> config = AmplifyConfig()
        >>> model = AmplifyForMaskedLM(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 11, 37])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    # AMPLIFY does NOT tie the input/output embeddings: ``encoder.weight`` and
    # ``decoder.weight`` are independently learned in the upstream checkpoint.
    _tied_weights_keys = {
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        tied_weights = super().get_expanded_tied_weights_keys(all_submodels=all_submodels)
        if all_submodels:
            return tied_weights
        return tied_weights | self._tied_weights_keys

    def __init__(self, config: AmplifyConfig):
        super().__init__(config)
        self.model = AmplifyModel(config, add_pooling_layer=False)
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


class AmplifyForPreTraining(AmplifyForMaskedLM):
    pass


class AmplifyEmbeddings(nn.Module):
    """Token embeddings with optional post-embedding normalization."""

    def __init__(self, config: AmplifyConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.layer_norm_after_embedding:
            self.layer_norm = (
                nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
                if config.rms_norm
                else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            )
        else:
            self.layer_norm = None

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
    ) -> Tensor | NestedTensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.layer_norm is not None:
            inputs_embeds = self.layer_norm(inputs_embeds)
        return inputs_embeds


class AmplifyEncoder(nn.Module):
    def __init__(self, config: AmplifyConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AmplifyLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        if config.layer_norm_before_last_layer:
            self.layer_norm = (
                nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
                if config.rms_norm
                else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            )
        else:
            self.layer_norm = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                **kwargs,
            )
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


class AmplifyLayer(GradientCheckpointingLayer):
    def __init__(self, config: AmplifyConfig, layer_idx: int | None = None):
        super().__init__()
        self.chunk_size_feed_forward = getattr(config, "chunk_size_feed_forward", 0)
        self.seq_len_dim = 1
        self.attention = AmplifyAttention(config, layer_idx=layer_idx)
        self.layer_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.intermediate = AmplifyIntermediate(config)
        self.output = AmplifyOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        attention_output, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        attention_output_ln = self.layer_norm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class AmplifyAttention(nn.Module):
    def __init__(self, config: AmplifyConfig, layer_idx: int | None = None):
        super().__init__()
        self.layer_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.self = AmplifySelfAttention(config, layer_idx=layer_idx)
        self.output = AmplifySelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states_ln = self.layer_norm(hidden_states)
        attention_output, attn_weights = self.self(
            hidden_states_ln,
            attention_mask=attention_mask,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


class AmplifySelfAttention(nn.Module):
    """Multi-headed self-attention with adjacent-pair rotary position embeddings."""

    def __init__(self, config: AmplifyConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.rotary_embeddings = RotaryEmbedding(embedding_dim=self.head_dim, interleaved=True)

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

        attention_bias = attention_mask
        if isinstance(hidden_states, NestedTensor):
            attention_bias = None

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_bias,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, self.all_head_size).contiguous()
        return attn_output, attn_weights


class AmplifySelfOutput(nn.Module):
    def __init__(self, config: AmplifyConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class AmplifyIntermediate(nn.Module):
    """SwiGLU feed-forward block.

    Implements ``silu(gate_proj(x)) * up_proj(x)`` with an intermediate
    dimension reduced to ``2 * intermediate_size / 3`` rounded up to the next
    multiple of 8, matching the upstream checkpoint layout.
    """

    def __init__(self, config: AmplifyConfig):
        super().__init__()
        intermediate_size = _swiglu_intermediate_size(config.intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.feedforward_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.feedforward_bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return gate * up


class AmplifyOutput(nn.Module):
    def __init__(self, config: AmplifyConfig):
        super().__init__()
        intermediate_size = _swiglu_intermediate_size(config.intermediate_size)
        self.dense = nn.Linear(intermediate_size, config.hidden_size, bias=config.feedforward_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class AmplifyPooler(nn.Module):
    def __init__(self, config: AmplifyConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return self.activation(pooled_output)


def _swiglu_intermediate_size(intermediate_size: int, multiple_of: int = 8) -> int:
    """Reduce ``intermediate_size`` to ``2/3`` and round up to the next ``multiple_of``."""
    reduced = int(2 * intermediate_size / 3)
    return multiple_of * ((reduced + multiple_of - 1) // multiple_of)


AmplifyPreTrainedModel._can_record_outputs = {
    "hidden_states": AmplifyLayer,
    "attentions": OutputRecorder(AmplifyAttention, index=1, layer_name="attention"),
}
