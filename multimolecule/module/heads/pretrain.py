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
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from multimolecule.models.configuration_utils import MaskedLMHeadConfig, PreTrainedConfig

from .output import HeadOutput
from .registry import HeadRegistry
from .transform import HeadTransformRegistryHF


@HeadRegistry.register("masked_lm")
class MaskedLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(
        self, config: PreTrainedConfig, weight: Tensor | None = None, head_config: MaskedLMHeadConfig | None = None
    ):
        super().__init__()
        if head_config is None:
            head_config = config.lm_head if hasattr(config, "lm_head") else config.head  # type: ignore[assignment]
        self.config: MaskedLMHeadConfig = head_config  # type: ignore[assignment]
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        self.num_labels = config.vocab_size
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransformRegistryHF.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=False)
        if weight is not None:
            self.decoder.weight = weight
        if self.config.bias:
            self.bias = nn.Parameter(torch.zeros(self.num_labels))
            self.decoder.bias = self.bias
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...], labels: Tensor | None = None) -> HeadOutput:
        sequence_output = outputs[0]
        output = self.dropout(sequence_output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            return HeadOutput(output, F.cross_entropy(output.view(-1, self.num_labels), labels.view(-1)))
        return HeadOutput(output)
