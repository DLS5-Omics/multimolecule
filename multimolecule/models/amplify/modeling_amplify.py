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
from typing import Any, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from danling import NestedTensor
from torch import Tensor, nn
from transformers import initialization as init
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
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
from .configuration_amplify import AMPLIFYConfig


class AMPLIFYPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AMPLIFYConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["AMPLIFYLayer"]

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


class AMPLIFYModel(AMPLIFYPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AMPLIFYConfig, AMPLIFYModel, ProteinTokenizer
        >>> config = AMPLIFYConfig()
        >>> model = AMPLIFYModel(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 11, 640])
        >>> output["pooler_output"].shape
        torch.Size([1, 640])
    """

    def __init__(self, config: AMPLIFYConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = AMPLIFYEmbeddings(config)
        self.encoder = AMPLIFYEncoder(config)
        self.pooler = AMPLIFYPooler(config) if add_pooling_layer else None

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
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
            config=self.config, input_embeds=embedding_output, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(embedding_output, attention_mask=attn_mask, **kwargs)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class AMPLIFYForSequencePrediction(AMPLIFYPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AMPLIFYConfig, AMPLIFYForSequencePrediction, ProteinTokenizer
        >>> config = AMPLIFYConfig()
        >>> model = AMPLIFYForSequencePrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AMPLIFYConfig):
        super().__init__(config)
        self.model = AMPLIFYModel(config)
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


class AMPLIFYForTokenPrediction(AMPLIFYPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AMPLIFYConfig, AMPLIFYForTokenPrediction, ProteinTokenizer
        >>> config = AMPLIFYConfig()
        >>> model = AMPLIFYForTokenPrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 9)))
        >>> output["logits"].shape
        torch.Size([1, 9, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AMPLIFYConfig):
        super().__init__(config)
        self.model = AMPLIFYModel(config, add_pooling_layer=False)
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


class AMPLIFYForContactPrediction(AMPLIFYPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AMPLIFYConfig, AMPLIFYForContactPrediction, ProteinTokenizer
        >>> config = AMPLIFYConfig()
        >>> model = AMPLIFYForContactPrediction(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 9, 9)))
        >>> output["logits"].shape
        torch.Size([1, 9, 9, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AMPLIFYConfig):
        super().__init__(config)
        self.model = AMPLIFYModel(config, add_pooling_layer=False)
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


class AMPLIFYForMaskedLM(AMPLIFYPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AMPLIFYConfig, AMPLIFYForMaskedLM, ProteinTokenizer
        >>> config = AMPLIFYConfig()
        >>> model = AMPLIFYForMaskedLM(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 11, 27])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    # AMPLIFY does NOT tie the input/output embeddings: ``encoder.weight`` and
    # ``decoder.weight`` are independently learned in the upstream checkpoint.
    _tied_weights_keys = {
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config: AMPLIFYConfig):
        super().__init__(config)
        self.model = AMPLIFYModel(config, add_pooling_layer=False)
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
    ) -> Tuple[Tensor, ...] | MaskedLMOutput:
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


class AMPLIFYForPreTraining(AMPLIFYForMaskedLM):
    pass


class AMPLIFYEmbeddings(nn.Module):
    """Token embeddings with optional post-embedding normalization."""

    def __init__(self, config: AMPLIFYConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.layer_norm_after_embedding:
            self.layer_norm = (
                AMPLIFYRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
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


class AMPLIFYEncoder(nn.Module):
    def __init__(self, config: AMPLIFYConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_embeddings = AMPLIFYRotaryEmbedding(self.head_dim, config.max_position_embeddings)
        self.layer = nn.ModuleList([AMPLIFYLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        if config.layer_norm_before_last_layer:
            self.layer_norm = (
                AMPLIFYRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
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
    ) -> Tensor:
        seq_length = hidden_states.shape[1]
        freqs_cis = self.rotary_embeddings(seq_length, device=hidden_states.device)
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                **kwargs,
            )
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class AMPLIFYLayer(GradientCheckpointingLayer):
    """Pre-norm transformer encoder layer with rotary attention and SwiGLU feed-forward."""

    def __init__(self, config: AMPLIFYConfig, layer_idx: int | None = None):
        super().__init__()
        self.attention_norm = (
            AMPLIFYRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.attention = AMPLIFYAttention(config, layer_idx=layer_idx)
        self.ffn_norm = (
            AMPLIFYRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.mlp = AMPLIFYMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        attn_output, _ = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ffn_norm(hidden_states))
        return hidden_states


class AMPLIFYAttention(nn.Module):
    """Multi-headed self-attention with complex-cis rotary position embeddings."""

    def __init__(self, config: AMPLIFYConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=config.attention_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if freqs_cis is not None:
            query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

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
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, self.all_head_size).contiguous()
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output, attn_weights


class AMPLIFYMLP(nn.Module):
    """SwiGLU feed-forward block.

    Implements ``down_proj(silu(gate_proj(x)) * up_proj(x))`` with an intermediate
    dimension reduced to ``2 * intermediate_size / 3`` rounded up to the next
    multiple of 8, matching the upstream checkpoint layout.
    """

    def __init__(self, config: AMPLIFYConfig):
        super().__init__()
        intermediate_size = _swiglu_intermediate_size(config.intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.feedforward_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.feedforward_bias)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=config.feedforward_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.dropout(self.down_proj(gate * up))


class AMPLIFYRMSNorm(nn.Module):
    """RMSNorm with float32 statistics (matches AMPLIFY's float-cast behavior)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class AMPLIFYRotaryEmbedding(nn.Module):
    """Complex-valued ("cis") rotary position embeddings used by AMPLIFY.

    The frequencies are stored as a non-persistent complex64 buffer and rebuilt
    on the first forward pass after a ``meta``-device materialisation.
    """

    _is_hf_initialized = True

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("freqs_cis", _precompute_freqs_cis(head_dim, max_position_embeddings, base, device=device),
                             persistent=False)
        self._initialized = False

    @torch.no_grad()
    def forward(self, seq_length: int, device: torch.device) -> Tensor:
        # Re-register the buffer on first real forward, after a meta-device init
        # has dropped the original complex tensor.
        if not self._initialized or self.freqs_cis.device != device or self.freqs_cis.shape[0] < seq_length:
            target_length = max(seq_length, self.max_position_embeddings)
            self.register_buffer(
                "freqs_cis",
                _precompute_freqs_cis(self.head_dim, target_length, self.base, device=device),
                persistent=False,
            )
            self._initialized = True
        return self.freqs_cis[:seq_length]


class AMPLIFYPooler(nn.Module):
    def __init__(self, config: AMPLIFYConfig):
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


def _precompute_freqs_cis(head_dim: int, end: int, base: float, device: torch.device | None = None) -> Tensor:
    """Precompute the complex64 frequency tensor used by AMPLIFY's rotary embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(end, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64, shape (end, head_dim // 2)


def apply_rotary_emb(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply complex-cis rotary embeddings to query and key tensors.

    Args:
        q: Shape ``(batch, num_heads, seq_length, head_dim)``. May be a
            ``NestedTensor`` when variable-length inputs are used.
        k: Shape ``(batch, num_heads, seq_length, head_dim)``. May be a
            ``NestedTensor`` when variable-length inputs are used.
        freqs_cis: Complex tensor of shape ``(seq_length, head_dim // 2)``.
    """
    if isinstance(q, NestedTensor) or isinstance(k, NestedTensor):
        q_storage, k_storage = [], []
        for q_t, k_t in zip(q._storage, k._storage):  # type: ignore[union-attr]
            q_r, k_r = _apply_rotary_emb_dense(q_t.unsqueeze(0), k_t.unsqueeze(0), freqs_cis)
            q_storage.append(q_r.squeeze(0))
            k_storage.append(k_r.squeeze(0))
        return (
            NestedTensor(q_storage, **q._meta()),  # type: ignore[union-attr]
            NestedTensor(k_storage, **k._meta()),  # type: ignore[union-attr]
        )
    return _apply_rotary_emb_dense(q, k, freqs_cis)


def _apply_rotary_emb_dense(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    seq_length = q.shape[-2]
    freqs_cis = freqs_cis[:seq_length]
    # Reshape (..., head_dim) -> (..., head_dim // 2, 2) and view as complex
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    # Broadcast freqs_cis over leading dims
    while freqs_cis.ndim < q_complex.ndim:
        freqs_cis = freqs_cis.unsqueeze(0)
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


AMPLIFYPreTrainedModel._can_record_outputs = {
    "hidden_states": AMPLIFYLayer,
    "attentions": OutputRecorder(AMPLIFYAttention, index=1),
}
