from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers.activations import ACT2FN

from multimolecule.models.configuration_utils import PretrainedConfig

from ..criterions import Criterion
from .output import HeadOutput
from .transform import HeadTransforms
from .utils import average_product_correct, symmetrize


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
        self.transform = HeadTransforms.build(self.config)
        self.decoder = nn.Linear(
            config.num_hidden_layers * config.num_attention_heads, self.num_labels, bias=self.config.bias
        )
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = Criterion(self.config)

    def forward(
        self,
        attentions: Tensor,
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
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
        if labels is not None:
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)
