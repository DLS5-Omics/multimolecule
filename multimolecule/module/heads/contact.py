# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput

from multimolecule.models.configuration_utils import HeadConfig, PreTrainedConfig

from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .utils import average_product_correct, symmetrize


@HeadRegistry.register("contact")
class ContactPredictionHead(PredictionHead):
    """
    Head for contact-map-level tasks.
    Performs symmetrization, and average product correct.
    """

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.decoder = nn.Linear(
            config.num_hidden_layers * config.num_attention_heads, self.num_labels, bias=self.config.bias
        )

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        attentions = torch.stack(outputs[-1], 1)
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
        attentions = attentions.permute(0, 2, 3, 1).squeeze(3)

        return super().forward(attentions, labels)
