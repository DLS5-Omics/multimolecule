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

from torch import Tensor, nn
from transformers.activations import ACT2FN

from multimolecule.models.configuration_utils import HeadConfig, PreTrainedConfig

from ..criterions import Criterion
from .output import HeadOutput
from .transform import HeadTransformRegistryHF


class PredictionHead(nn.Module):
    """Head for all-level of tasks."""

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
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = Criterion(self.config)

    def forward(self, embeddings: Tensor, labels: Tensor | None) -> HeadOutput:
        output = self.dropout(embeddings)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)
