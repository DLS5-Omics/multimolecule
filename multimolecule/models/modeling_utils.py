from __future__ import annotations

from typing import Optional, Tuple

import torch
from chanfig import ConfigRegistry
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from .configuration_utils import HeadConfig, PretrainedConfig


class ContactPredictionHead(nn.Module):
    """
    Head for contact-map-level tasks.
    Performs symmetrization, and average product correct.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config.head
        self.num_labels = config.head.num_labels
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = PredictionHeadTransform.build(self.config)
        self.decoder = nn.Linear(
            config.num_hidden_layers * config.num_attention_heads, self.num_labels, bias=self.config.bias
        )
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None

    def forward(
        self, attentions: Tensor, attention_mask: Optional[Tensor] = None, input_ids: Optional[Tensor] = None
    ) -> Tensor:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError(
                    "Either attention_mask or input_ids must be provided for contact prediction head to work."
                )
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to contact prediction head."
                )
            attention_mask = input_ids.ne(self.pad_token_id)
        # In the original model, attentions for padding tokens are completely zeroed out.
        # This makes no difference most of the time because the other tokens won't attend to them,
        # but it does for the contact prediction task, which takes attentions as input,
        # so we have to mimic that here.
        attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        attentions *= attention_mask[:, None, None, :, :]
        # remove cls token attentions
        if self.bos_token_id is not None:
            attentions = attentions[..., 1:, 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token attentions
        if self.eos_token_id is not None:
            if input_ids is not None:
                # Zhiyuan: Do we really need to remove the eos token attentions?
                eos_mask = input_ids.ne(self.eos_token_id).to(attentions)
                eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
                attentions *= eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]

        # features: batch x channels x input_ids x input_ids (symmetric)
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
        attentions = attentions.to(self.decoder.weight.device)
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        output = self.dropout(attentions)
        output = self.decoder(output).squeeze(3)
        if self.activation is not None:
            output = self.activation(output)
        return output


class MaskedLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, config: PretrainedConfig, weight: Optional[Tensor] = None):
        super().__init__()
        self.config = config.lm_head if hasattr(config, "lm_head") else config.head
        self.num_labels = config.vocab_size
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = PredictionHeadTransform.build(self.config)

        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        if weight is not None:
            self.decoder.weight = weight
        if self.config.bias:
            self.bias = nn.Parameter(torch.zeros(self.num_labels))
            self.decoder.bias = self.bias
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...]) -> Tensor:
        sequence_output = outputs[0]
        output = self.dropout(sequence_output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class SequenceClassificationHead(nn.Module):
    """Head for sequence-level tasks."""

    num_labels: int

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config.head
        self.num_labels = config.head.num_labels
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = PredictionHeadTransform.build(self.config)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...]) -> Tensor:
        sequence_output = outputs[1]
        output = self.dropout(sequence_output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class TokenClassificationHead(nn.Module):
    """Head for token-level tasks."""

    num_labels: int

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config.head
        self.num_labels = config.head.num_labels
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = PredictionHeadTransform.build(self.config)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...]) -> Tensor:
        token_output = outputs[0]
        output = self.dropout(token_output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


PredictionHeadTransform = ConfigRegistry(key="transform")


@PredictionHeadTransform.register("nonlinear")
class NonLinearTransform(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.transform_act, str):
            self.transform_act_fn = ACT2FN[config.transform_act]
        else:
            self.transform_act_fn = config.transform_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@PredictionHeadTransform.register("linear")
class LinearTransform(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@PredictionHeadTransform.register(None)
class IdentityTransform(nn.Identity):
    def __init__(self, config: HeadConfig):  # pylint: disable=unused-argument
        super().__init__()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def average_product_correct(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized
