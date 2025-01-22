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

from typing import TYPE_CHECKING, Tuple
from warnings import warn

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.activations import ACT2FN

from ..criterions import CriterionRegistry
from .config import HeadConfig
from .output import HeadOutput
from .transform import HeadTransformRegistryHF

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


class PredictionHead(nn.Module):
    r"""
    Head for all-level of tasks.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    num_labels: int
    require_attentions: bool = False

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__()
        if head_config is None:
            head_config = config.head or HeadConfig(num_labels=config.num_labels)
        elif head_config.num_labels is None:
            head_config.num_labels = config.num_labels
        self.config = head_config
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        if self.config.problem_type is None:
            self.config.problem_type = config.problem_type
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.num_labels = self.config.num_labels  # type: ignore[assignment]
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransformRegistryHF.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = CriterionRegistry.build(self.config)

    def forward(self, embeddings: Tensor, labels: Tensor | None, **kwargs) -> HeadOutput:
        r"""
        Forward pass of the PredictionHead.

        Args:
            embeddings: The embeddings to be passed through the head.
            labels: The labels for the head.
        """
        if kwargs:
            warn(
                f"The following arguments are not applicable to {self.__class__.__name__}"
                f"and will be ignored: {kwargs.keys()}"
            )
        output = self.dropout(embeddings)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(output, Tensor):
                    output = labels.nested_like(output, strict=False)
                return HeadOutput(output, self.criterion(output.concat, labels.concat))
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)

    def _get_attention_mask(self, input_ids: NestedTensor | Tensor) -> Tensor:
        if isinstance(input_ids, NestedTensor):
            return input_ids.mask
        if input_ids is None:
            raise ValueError(
                f"Either attention_mask or input_ids must be provided for {self.__class__.__name__} to work."
            )
        if self.pad_token_id is None:
            raise ValueError(
                f"pad_token_id must be provided when attention_mask is not passed to {self.__class__.__name__}."
            )
        return input_ids.ne(self.pad_token_id).int()

    def _remove_special_tokens(
        self, output: Tensor, attention_mask: Tensor, input_ids: Tensor | None
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
                input_ids = input_ids[..., :-1]
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) == last_valid_indices.unsqueeze(1)
            output = output * eos_mask[:, :, None]
            output = output[..., :-1, :]
            attention_mask = attention_mask[..., 1:]
        return output, attention_mask, input_ids
