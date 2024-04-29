from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import torch
from chanfig import ConfigRegistry
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from .configuration_utils import HeadConfig, PretrainedConfig

TokenHeads = ConfigRegistry(key="tokenizer_type")
NucleotideHeads = ConfigRegistry(key="tokenizer_type")


class ContactPredictionHead(nn.Module):
    """
    Head for contact-map-level tasks.
    Performs symmetrization, and average product correct.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config.head
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        if self.config.num_labels is None:
            self.config.num_labels = config.num_labels
        if self.config.problem_type is None:
            self.config.problem_type = config.problem_type
        self.num_labels = self.config.num_labels
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
                    "Either attention_mask or input_ids must be provided for ContactPredictionHead to work."
                )
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to ContactPredictionHead."
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
            attention_mask = attention_mask[..., 1:, 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token attentions
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(attentions)
                input_ids = input_ids[..., 1:]
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=attentions.device).unsqueeze(0) == last_valid_indices
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions *= eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
            attention_mask = attention_mask[..., 1:, 1:]

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
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        self.num_labels = config.vocab_size
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = PredictionHeadTransform.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=False)
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


class ClassificationHead(nn.Module):
    """Head for all-level of tasks."""

    num_labels: int

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config.head
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        if self.config.num_labels is None:
            self.config.num_labels = config.num_labels
        if self.config.problem_type is None:
            self.config.problem_type = config.problem_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = PredictionHeadTransform.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None

    def forward(self, embeddings: Tensor) -> Tensor:
        output = self.dropout(embeddings)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class SequenceClassificationHead(ClassificationHead):
    """Head for sequence-level tasks."""

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...]) -> Tensor:  # pylint: disable=arguments-renamed
        output = super().forward(outputs[1])
        return output


@TokenHeads.register("single", default=True)
class TokenClassificationHead(ClassificationHead):
    """Head for token-level tasks."""

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...]) -> Tensor:  # pylint: disable=arguments-renamed
        output = super().forward(outputs[0])
        return output


@TokenHeads.register("kmer")
class TokenKMerHead(ClassificationHead):
    """Head for token-level tasks."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.nmers = config.nmers
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.unfold_kmer_embeddings = partial(
            unfold_kmer_embeddings, nmers=self.nmers, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
        )

    def forward(  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided for TokenKMerHead to work.")
            if self.pad_token_id is None:
                raise ValueError("pad_token_id must be provided when attention_mask is not passed to TokenKMerHead.")
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0]
        output = self.unfold_kmer_embeddings(output, attention_mask)
        output = super().forward(output)
        return output


@NucleotideHeads.register("single", default=True)
class NucleotideClassificationHead(ClassificationHead):
    """Head for nucleotide-level tasks."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

    def forward(  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError(
                    "Either attention_mask or input_ids must be provided for NucleotideClassificationHead to work."
                )
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to NucleotideClassificationHead."
                )
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0]
        # remove cls token embeddings
        if self.bos_token_id is not None:
            output = output[..., 1:, :]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token embeddings
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output)
                input_ids = input_ids[..., 1:]
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) == last_valid_indices.unsqueeze(1)
            output *= eos_mask[:, :, None]
            output = output[..., :-1, :]
            attention_mask = attention_mask[..., 1:]

        output = super().forward(output)
        return output


@NucleotideHeads.register("kmer")
class NucleotideKMerHead(ClassificationHead):
    """Head for nucleotide-level tasks."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.nmers = config.nmers
        self.bos_token_id = None  # Nucleotide-level head removes <cls> token.
        self.eos_token_id = None  # Nucleotide-level head removes <eos> token.
        self.pad_token_id = config.pad_token_id
        self.unfold_kmer_embeddings = partial(unfold_kmer_embeddings, nmers=self.nmers)

    def forward(  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided for NucleotideKMerHead to work.")
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to NucleotideKMerHead."
                )
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0]
        # remove cls token embeddings
        if self.bos_token_id is not None:
            output = output[..., 1:, :]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token embeddings
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output)
                input_ids = input_ids[..., 1:]
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) == last_valid_indices.unsqueeze(1)
            output *= eos_mask[:, :, None]
            output = output[..., :-1, :]
            attention_mask = attention_mask[..., 1:]

        output = self.unfold_kmer_embeddings(output, attention_mask)
        output = super().forward(output)
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


def unfold_kmer_embeddings(
    embeddings: Tensor,
    attention_mask: Tensor,
    nmers: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    r"""
    Unfold k-mer embeddings to token embeddings.

    For k-mer input, each embedding column represents k tokens.
    This should be fine for sequence level tasks, but sacrifices the resolution for token level tasks.
    This function unfolds the k-mer embeddings to token embeddings by sliding averaging the k-mer embeddings.

    For example:

    input tokens = `ACGU`

    2-mer embeddings = `[<CLS>, AC, CG, GU, <SEP>]`.

    token embeddings = `[<CLS>, AC, (AC + CG) / 2, (CG + GU) / 2, GU, <SEP>]`.

    Args:
        embeddings: The k-mer embeddings.
        attention_mask: The attention mask.
        nmers: The number of tokens in each k-mer.
        bos_token_id: The id of the beginning of sequence token.
            If not None, the first valid token will not be included in sliding averaging.
        eos_token_id: The id of the end of sequence token.
            If not None, the last valid token will not be included in sliding averaging.

    Returns:
        The token embeddings.

    Examples:
        >>> from danling import NestedTensor
        >>> embeddings = NestedTensor(torch.arange(3).repeat(2, 1).T, torch.arange(5).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 3, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.0, 2.0, 3.0, 0.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        >>> embeddings = NestedTensor(torch.arange(5).repeat(2, 1).T, torch.arange(7).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 4, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 5.0, 0.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5, 6.0, 7.0]
        >>> embeddings = NestedTensor(torch.arange(7).repeat(2, 1).T, torch.arange(11).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 5, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0]
        >>> embeddings = NestedTensor(torch.arange(3).repeat(2, 1).T, torch.arange(4).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 6, True, True)
        >>> output[0, :, 0].tolist()
        [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 4.0]
        >>> embeddings = NestedTensor(torch.arange(1).repeat(2, 1).T, torch.arange(2).repeat(2, 1).T) + 1
        >>> output = unfold_kmer_embeddings(embeddings.tensor.float(), embeddings.mask, 6)
        >>> output[0, :, 0].tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        >>> output[1, :, 0].tolist()
        [1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0]
    """

    batch_size, seq_length, hidden_size = embeddings.size()
    last_valid_indices = attention_mask.sum(dim=-1)
    output = torch.zeros(batch_size, seq_length + nmers - 1, hidden_size, device=embeddings.device)
    for index, (tensor, seq_len) in enumerate(zip(embeddings, last_valid_indices)):
        embedding = tensor[:seq_len]
        if bos_token_id is not None:
            embedding = embedding[1:]
        if eos_token_id is not None:
            embedding = embedding[:-1]
        if len(embedding) > nmers:
            begin = torch.stack([embedding[:i].mean(0) for i in range(1, nmers)])
            medium = embedding.unfold(0, nmers, 1).mean(-1)
            end = torch.stack([embedding[-i:].mean(0) for i in range(nmers - 1, 0, -1)])
            embedding = torch.cat([begin, medium, end])
        elif len(embedding) > 2:
            begin = torch.stack([embedding[:i].mean(0) for i in range(1, len(embedding))])
            end = torch.stack([embedding[-i:].mean(0) for i in range(nmers, 0, -1)])
            embedding = torch.cat([begin, end])
        elif len(embedding) == 2:
            medium = embedding.mean(0).repeat(nmers - 1, 1)
            embedding = torch.cat([embedding[0][None, :], medium, embedding[1][None, :]])
        elif len(embedding) == 1:
            embedding = embedding.repeat(nmers, 1)
        else:
            raise ValueError("Sequence length is less than nmers.")
        if bos_token_id is not None:
            embedding = torch.cat([tensor[0][None, :], embedding])
        if eos_token_id is not None:
            embedding = torch.cat([embedding, tensor[seq_len - 1][None, :]])
        output[index, : seq_len + nmers - 1] = embedding
    return output


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
