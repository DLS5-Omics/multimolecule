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

from typing import TYPE_CHECKING, Mapping, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from .config import MaskedLMHeadConfig
from .output import HeadOutput
from .registry import HeadRegistry
from .transform import HeadTransformRegistryHF

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


@HeadRegistry.register("masked_lm")
class MaskedLMHead(nn.Module):
    r"""
    Head for masked language modeling.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "last_hidden_state"
    r"""The default output to use for the head."""

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
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name

    def forward(
        self, outputs: ModelOutput | Tuple[Tensor, ...], labels: Tensor | None = None, output_name: str | None = None
    ) -> HeadOutput:
        r"""
        Forward pass of the MaskedLMHead.

        Args:
            outputs: The outputs of the model.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        output = self.dropout(output)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(output, Tensor):
                    output = labels.nested_like(output, strict=False)
                return HeadOutput(output, F.cross_entropy(torch.cat(output.storage()), torch.cat(labels.storage())))
            return HeadOutput(output, F.cross_entropy(output.view(-1, self.num_labels), labels.view(-1)))
        return HeadOutput(output)
