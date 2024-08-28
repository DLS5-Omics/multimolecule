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

from typing import TYPE_CHECKING
from warnings import warn

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.activations import ACT2FN

from ..criterions import Criterion
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

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__()
        if head_config is None:
            head_config = config.head
        self.config = head_config
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        if self.config.num_labels is None:
            self.config.num_labels = config.num_labels
        if self.config.problem_type is None:
            self.config.problem_type = config.problem_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransformRegistryHF.build(self.config)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = Criterion(self.config)

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
                return HeadOutput(output, self.criterion(torch.cat(output.storage()), torch.cat(labels.storage())))
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)
