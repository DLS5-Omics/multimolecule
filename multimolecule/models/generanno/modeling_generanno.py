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

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple

from multimolecule.modules import (
    ContactPredictionHead,
    MaskedLMHead,
    MaskedLMHeadConfig,
    SequencePredictionHead,
    TokenPredictionHead,
    eager_attention_forward,
)

from ..modeling_outputs import ContactPredictorOutput, SequencePredictorOutput, TokenPredictorOutput
from .configuration_generanno import GenerannoConfig


class GenerannoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GenerannoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["GenerannoLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, GenerannoRMSNorm):
            init.ones_(module.weight)


class GenerannoModel(GenerannoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import GenerannoConfig, GenerannoModel, DnaTokenizer
        >>> config = GenerannoConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
        ...                          num_attention_heads=4, num_key_value_heads=2,
        ...                          max_position_embeddings=64)
        >>> model = GenerannoModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/dna")
        >>> input = tokenizer("ACGTACGT", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 10, 32])
        >>> output["pooler_output"].shape
        torch.Size([1, 32])
    """

    def __init__(self, config: GenerannoConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = GenerannoEmbeddings(config)
        self.encoder = GenerannoEncoder(config)
        self.rotary_emb = GenerannoRotaryEmbedding(config)
        self.pooler = GenerannoPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if isinstance(input_ids, NestedTensor) and attention_mask is None:
            attention_mask = input_ids.mask
        if isinstance(inputs_embeds, NestedTensor) and attention_mask is None:
            attention_mask = inputs_embeds.mask
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if attention_mask is None and input_ids is not None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        embedding_output = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if position_ids is None:
            seq_length = embedding_output.shape[1]
            position_ids = torch.arange(seq_length, device=embedding_output.device).unsqueeze(0)

        causal_mask = create_bidirectional_mask(
            config=self.config, input_embeds=embedding_output, attention_mask=attention_mask
        )

        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class GenerannoForSequencePrediction(GenerannoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import GenerannoConfig, GenerannoForSequencePrediction, DnaTokenizer
        >>> config = GenerannoConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
        ...                          num_attention_heads=4, num_key_value_heads=2,
        ...                          max_position_embeddings=64)
        >>> model = GenerannoForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/dna")
        >>> input = tokenizer("ACGTACGT", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: GenerannoConfig):
        super().__init__(config)
        self.model = GenerannoModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
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


class GenerannoForTokenPrediction(GenerannoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import GenerannoConfig, GenerannoForTokenPrediction, DnaTokenizer
        >>> config = GenerannoConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
        ...                          num_attention_heads=4, num_key_value_heads=2,
        ...                          max_position_embeddings=64)
        >>> model = GenerannoForTokenPrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/dna")
        >>> input = tokenizer("ACGTACGT", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 8)))
        >>> output["logits"].shape
        torch.Size([1, 8, 1])
    """

    def __init__(self, config: GenerannoConfig):
        super().__init__(config)
        self.model = GenerannoModel(config, add_pooling_layer=False)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | TokenPredictorOutput:
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


class GenerannoForContactPrediction(GenerannoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import GenerannoConfig, GenerannoForContactPrediction, DnaTokenizer
        >>> config = GenerannoConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
        ...                          num_attention_heads=4, num_key_value_heads=2,
        ...                          max_position_embeddings=64)
        >>> model = GenerannoForContactPrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/dna")
        >>> input = tokenizer("ACGTACGT", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 8, 8)))
        >>> output["logits"].shape
        torch.Size([1, 8, 8, 1])
    """

    def __init__(self, config: GenerannoConfig):
        super().__init__(config)
        self.model = GenerannoModel(config, add_pooling_layer=False)
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config
        self.require_attentions = self.contact_head.require_attentions

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ContactPredictorOutput:
        if self.require_attentions:
            kwargs["output_attentions"] = True
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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


class GenerannoForMaskedLM(GenerannoPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import GenerannoConfig, GenerannoForMaskedLM, DnaTokenizer
        >>> config = GenerannoConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
        ...                          num_attention_heads=4, num_key_value_heads=2,
        ...                          max_position_embeddings=64)
        >>> model = GenerannoForMaskedLM(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/dna")
        >>> input = tokenizer("ACGTACGT", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 10, 26])
    """

    def __init__(self, config: GenerannoConfig):
        super().__init__(config)
        self.model = GenerannoModel(config, add_pooling_layer=False)
        # The original GENERanno checkpoints store an untied `lm_head` with no transform and no bias.
        # We force the head config so the converted state dict loads with neither a transform nor a bias.
        lm_head_config = config.lm_head or MaskedLMHeadConfig()
        lm_head_config.transform = "identity"
        lm_head_config.bias = False
        config.lm_head = lm_head_config
        self.lm_head = MaskedLMHead(config, head_config=lm_head_config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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


class GenerannoForPreTraining(GenerannoForMaskedLM):
    pass


class GenerannoEmbeddings(nn.Module):
    def __init__(self, config: GenerannoConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return inputs_embeds


class GenerannoEncoder(nn.Module):
    def __init__(self, config: GenerannoConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GenerannoLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.layer_norm = GenerannoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_embeddings: Tuple[Tensor, Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class GenerannoLayer(GradientCheckpointingLayer):
    def __init__(self, config: GenerannoConfig, layer_idx: int | None = None):
        super().__init__()
        self.input_layer_norm = GenerannoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GenerannoAttention(config, layer_idx=layer_idx)
        self.post_attention_layer_norm = GenerannoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GenerannoMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_embeddings: Tuple[Tensor, Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GenerannoAttention(nn.Module):
    def __init__(self, config: GenerannoConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.query = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.key = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.value = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_embeddings: Tuple[Tensor, Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, Tensor | None]:
        bsz, q_len, _ = hidden_states.shape

        query_states = (
            self.query(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        )
        key_states = self.key(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = (
            self.value(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        if position_embeddings is None:
            raise ValueError("position_embeddings (cos, sin) must be provided.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.output(attn_output)
        return attn_output, attn_weights


class GenerannoMLP(nn.Module):
    """SwiGLU feed-forward block: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: GenerannoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class GenerannoRMSNorm(nn.Module):
    """RMSNorm: equivalent to T5LayerNorm. Computes the mean of squares along the last dim."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GenerannoRotaryEmbedding(nn.Module):
    """LLaMA-style rotary position embeddings.

    The `inv_freq` buffer is deterministic and reconstructable from config; it is registered non-persistent
    so it is excluded from the state dict, and re-registered on first forward to recover from meta-init.
    """

    _is_hf_initialized = True

    def __init__(self, config: GenerannoConfig, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.dim = config.hidden_size // config.num_attention_heads
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._initialized = False

    def _compute_inv_freq(self, device: torch.device | None) -> Tensor:
        exponent = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        return 1.0 / (self.base**exponent)

    @torch.no_grad()
    def forward(self, hidden_states: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        if not self._initialized or self.inv_freq.device != hidden_states.device:
            inv_freq = self._compute_inv_freq(hidden_states.device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._initialized = True

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = hidden_states.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


class GenerannoPooler(nn.Module):
    """Trainable pooler that emits a single vector per sequence from the first token's hidden state.

    The upstream GENERanno checkpoint stores no pooler weights; downstream classifiers in this repo
    train this small dense+tanh head from scratch alongside their task-specific output layers.
    """

    def __init__(self, config: GenerannoConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return self.activation(pooled_output)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input (cat-halves, not interleaved)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> Tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


GenerannoPreTrainedModel._can_record_outputs = {
    "hidden_states": GenerannoLayer,
    "attentions": OutputRecorder(GenerannoAttention, index=1, layer_name="attention"),
}
