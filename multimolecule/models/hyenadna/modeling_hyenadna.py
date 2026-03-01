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
from typing import Any, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import (
    SequencePredictionHead,
    TokenPredictionHead,
)

from ..modeling_outputs import SequencePredictorOutput, TokenPredictorOutput
from .configuration_hyenadna import HyenaDnaConfig


def fftconv(u: Tensor, k: Tensor, D: Tensor) -> Tensor:
    """FFT-based convolution via the Convolution Theorem.

    Computes ``y = conv(u, k) + u * D`` using FFT for O(L log L) complexity.

    Args:
        u: Input tensor of shape ``(batch, channels, seq_len)``.
        k: Filter kernel of shape ``(channels, seq_len)``.
        D: Bias/skip connection of shape ``(channels,)``.

    Returns:
        Output tensor of shape ``(batch, channels, seq_len)``.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=torch.float32), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class HyenaDnaSinActivation(nn.Module):
    """Sin activation function with learnable or fixed frequency for the Hyena implicit filter."""

    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        if config.train_freq:
            self.freq = nn.Parameter(config.activation_freq * torch.ones(1, config.filter_order))
        else:
            self.register_buffer("freq", config.activation_freq * torch.ones(1, config.filter_order))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.freq * x)


class HyenaDnaPositionalEmbedding(nn.Module):
    """Complex exponential positional embeddings for Hyena filters.

    Produces a ``(1, L, emb_dim)`` embedding from time ``t`` in ``[0, 1]``
    combined with sinusoidal frequency bands.
    """

    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        self.seq_len = config.max_position_embeddings
        # Normalized time in [0, 1]
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # (1, L, 1)

        if config.filter_emb_dim > 1:
            bands = (config.filter_emb_dim - 1) // 2
        # Integer positions for frequency computation
        t_rescaled = torch.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / self.seq_len  # (1, L, 1)

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]  # (1, 1, bands)

        z = torch.cat([t, torch.cos(-f * w), torch.sin(-f * w)], dim=-1)  # (1, L, emb_dim)

        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L: int) -> tuple[Tensor, Tensor]:
        return self.z[:, :L], self.t[:, :L]


class HyenaDnaExponentialModulation(nn.Module):
    """Exponential decay window applied to the output of the implicit filter MLP."""

    def __init__(
        self,
        d_model: int,
        fast_decay_pct: float = 0.3,
        slow_decay_pct: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.05,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        decay = torch.exp(-t * self.deltas.abs())
        return x * (decay + self.shift)


class HyenaDnaFilter(nn.Module):
    """Implicit long convolution filter parameterized by an MLP.

    The filter kernel is generated by feeding positional embeddings through
    an MLP with Sin activations, then applying exponential modulation.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        self.d_model = config.hidden_size * (config.hyena_order - 1)
        self.use_bias = config.use_bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(config.filter_dropout)

        act = HyenaDnaSinActivation(config)
        self.seq_len = config.max_position_embeddings

        self.pos_emb = HyenaDnaPositionalEmbedding(config)

        self.implicit_filter = nn.Sequential(
            nn.Linear(config.filter_emb_dim, config.filter_order),
            act,
        )
        for _ in range(config.num_inner_mlps):
            self.implicit_filter.append(nn.Linear(config.filter_order, config.filter_order))
            self.implicit_filter.append(HyenaDnaSinActivation(config))

        self.implicit_filter.append(nn.Linear(config.filter_order, config.hidden_size, bias=False))

        self.modulation = HyenaDnaExponentialModulation(config.hidden_size)

    def filter(self, L: int) -> Tensor:
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z.to(dtype=self.implicit_filter[0].weight.dtype))
        h = self.modulation(t, h)
        return h

    def forward(self, x: Tensor, L: int, k: Tensor | None = None, bias: Tensor | None = None) -> Tensor:
        if k is None:
            k = self.filter(L)
        k = k[0] if isinstance(k, tuple) else k
        y = fftconv(x, k, bias)
        return y


class HyenaDnaOperator(nn.Module):
    r"""Hyena operator: subquadratic drop-in replacement for attention.

    Implements the recurrence from `Hyena Hierarchy <https://arxiv.org/abs/2302.10866>`_:
    project input to ``(order + 1)`` channels, apply a short depthwise convolution,
    then iteratively gate and convolve with learned long-range filters.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.l_max = config.max_position_embeddings
        self.order = config.hyena_order
        inner_width = config.hidden_size * (self.order + 1)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.in_proj = nn.Linear(self.d_model, inner_width)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            config.short_filter_order,
            padding=config.short_filter_order - 1,
            groups=inner_width,
        )
        self.filter_fn = HyenaDnaFilter(config)

    def forward(self, u: Tensor) -> Tensor:
        seq_len = u.size(-2)
        l_filter = min(seq_len, self.l_max)
        u = self.in_proj(u).transpose(1, 2)

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
        bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = (v * x[0]).transpose(1, 2)
        y = self.out_proj(y)
        return y


class HyenaDnaMlp(nn.Module):
    """Feed-forward MLP with GELU activation."""

    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class HyenaDnaBlock(GradientCheckpointingLayer):
    """Single Hyena block: pre-norm Hyena mixer + pre-norm MLP, both with residual connections."""

    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        self.mixer = HyenaDnaOperator(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = HyenaDnaMlp(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        hidden_states = self.mixer(hidden_states)
        residual = hidden_states + residual

        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class HyenaDnaEmbeddings(nn.Module):
    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (vocab_size % config.pad_vocab_size_multiple)
        self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.word_embeddings(input_ids)


class HyenaDnaPooler(nn.Module):
    def __init__(self, config: HyenaDnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class HyenaDnaPreTrainedModel(PreTrainedModel):
    config_class = HyenaDnaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["HyenaDnaBlock"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class HyenaDnaModel(HyenaDnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import HyenaDnaConfig, HyenaDnaModel
        >>> config = HyenaDnaConfig()
        >>> model = HyenaDnaModel(config)
    """

    def __init__(self, config: HyenaDnaConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.embeddings = HyenaDnaEmbeddings(config)
        self.dropout = nn.Dropout(config.embedding_dropout)
        self.layers = nn.ModuleList([HyenaDnaBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = HyenaDnaPooler(config) if add_pooling_layer else None

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
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        if isinstance(input_ids, NestedTensor):
            input_ids = input_ids.tensor

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = self.dropout(inputs_embeds)
        all_hidden_states: tuple[Tensor, ...] = ()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states.to(dtype=self.final_layer_norm.weight.dtype))

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class HyenaDnaForCausalLM(HyenaDnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import HyenaDnaConfig, HyenaDnaForCausalLM
        >>> config = HyenaDnaConfig()
        >>> model = HyenaDnaForCausalLM(config)
    """

    def __init__(self, config: HyenaDnaConfig):
        super().__init__(config)
        self.model = HyenaDnaModel(config, add_pooling_layer=False)
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (vocab_size % config.pad_vocab_size_multiple)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

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
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class HyenaDnaForSequencePrediction(HyenaDnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import HyenaDnaConfig, HyenaDnaForSequencePrediction
        >>> config = HyenaDnaConfig()
        >>> model = HyenaDnaForSequencePrediction(config)
    """

    def __init__(self, config: HyenaDnaConfig):
        super().__init__(config)
        self.model = HyenaDnaModel(config)
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
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class HyenaDnaForTokenPrediction(HyenaDnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import HyenaDnaConfig, HyenaDnaForTokenPrediction
        >>> config = HyenaDnaConfig()
        >>> model = HyenaDnaForTokenPrediction(config)
    """

    def __init__(self, config: HyenaDnaConfig):
        super().__init__(config)
        self.model = HyenaDnaModel(config, add_pooling_layer=False)
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
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return TokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class HyenaDnaForPreTraining(HyenaDnaForCausalLM):
    pass


HyenaDnaPreTrainedModel._can_record_outputs = {
    "hidden_states": HyenaDnaBlock,
}
