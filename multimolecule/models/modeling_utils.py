from typing import Optional

import torch
from chanfig import Registry
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput


class ContactPredictionHead(nn.Module):
    """
    Head for contact-map-level tasks.
    Performs symmetrization, and average product correct.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        in_features: int,
        *,
        transform: str = "none",
        dropout: float = 0.0,
        activation: Optional[str] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(dropout)
        self.transform = PredictionHeadTransform.build(transform, config)
        self.decoder = nn.Linear(in_features, config.num_labels, bias=bias)
        self.activation = ACT2FN[activation] if activation is not None else None

    def forward(self, attentions: Tensor, input_ids: Tensor) -> Tensor:
        # remove cls token attentions
        if self.bos_token_id is not None:
            attentions = attentions[..., 1:, 1:]
        # remove eos token attentions
        if self.eos_token_id is not None:
            eos_mask = input_ids.ne(self.eos_token_id).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
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

    def __init__(
        self,
        config: PretrainedConfig,
        weight: Optional[Tensor] = None,
        *,
        transform: str = "nonlinear",
        dropout: float = 0.0,
        activation: Optional[str] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.num_labels = config.vocab_size
        self.dropout = nn.Dropout(dropout)
        self.transform = PredictionHeadTransform.build(transform, config)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=bias)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))
        if weight is not None:
            self.decoder.weight = weight
        self.decoder.bias = self.bias
        self.activation = ACT2FN[activation] if activation is not None else None

    def forward(self, outputs: ModelOutput) -> Tensor:
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

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        transform: str = "none",
        dropout: float = 0.0,
        activation: Optional[str] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(dropout)
        self.transform = PredictionHeadTransform.build(transform, config)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=bias)
        self.activation = ACT2FN[activation] if activation is not None else None

    def forward(self, outputs: ModelOutput) -> Tensor:
        sequence_output = outputs[0]
        output = self.dropout(sequence_output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class TokenClassificationHead(nn.Module):
    """Head for token-level tasks."""

    num_labels: int

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        transform: str = "none",
        dropout: float = 0.0,
        activation: Optional[str] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(dropout)
        self.transform = PredictionHeadTransform.build(transform, config)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=bias)
        self.activation = ACT2FN[activation] if activation is not None else None

    def forward(self, outputs: ModelOutput) -> Tensor:
        token_output = outputs[1]
        output = self.dropout(token_output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


PredictionHeadTransform = Registry()


@PredictionHeadTransform.register("nonlinear")
class NonLinearTransform(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@PredictionHeadTransform.register("linear")
class LinearTransform(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@PredictionHeadTransform.register("none")
class IdentityTransform(nn.Identity):
    def __init__(self, config: PretrainedConfig):  # pylint: disable=unused-argument
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
