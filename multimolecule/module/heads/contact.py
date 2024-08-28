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

from typing import Mapping, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput
from typing_extensions import TYPE_CHECKING

from .config import HeadConfig
from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .utils import average_product_correct, symmetrize

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


@HeadRegistry.register("contact")
class ContactPredictionHead(PredictionHead):
    r"""
    Head for tasks in contact-level.

    Performs symmetrization, and average product correct.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "attentions"
    r"""The default output to use for the head."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.decoder = nn.Linear(
            config.num_hidden_layers * config.num_attention_heads, self.num_labels, bias=self.config.bias
        )
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the ContactPredictionHead.

        Args:
            outputs: The outputs of the model.
            attention_mask: The attention mask for the inputs.
            input_ids: The input ids for the inputs.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if attention_mask is None:
            if isinstance(input_ids, NestedTensor):
                input_ids, attention_mask = input_ids.tensor, input_ids.mask
            else:
                if input_ids is None:
                    raise ValueError(
                        f"Either attention_mask or input_ids must be provided for {self.__class__.__name__} to work."
                    )
                if self.pad_token_id is None:
                    raise ValueError(
                        f"pad_token_id must be provided when attention_mask is not passed to {self.__class__.__name__}."
                    )
                attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[-1]
        attentions = torch.stack(output, 1)

        # In the original model, attentions for padding tokens are completely zeroed out.
        # This makes no difference most of the time because the other tokens won't attend to them,
        # but it does for the contact prediction task, which takes attentions as input,
        # so we have to mimic that here.
        attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        attentions *= attention_mask[:, None, None, :, :]

        # remove cls token attentions
        if self.bos_token_id is not None:
            attentions = attentions[..., 1:, 1:]
            # process attention_mask and input_ids to make removal of eos token happy
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token attentions
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(attentions)
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=attentions.device).unsqueeze(0) == last_valid_indices
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions *= eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]

        # features: batch x channels x input_ids x input_ids (symmetric)
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
        attentions = attentions.to(self.decoder.weight.device)
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1).squeeze(3)

        return super().forward(attentions, labels, **kwargs)
