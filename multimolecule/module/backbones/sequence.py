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

import torch
from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn

from .registry import BackboneRegistry
from .sequences import SequenceRegistry


@BackboneRegistry.register("sequence", default=True)
class SequenceBackbone(nn.Module):
    def __init__(self, sequence) -> None:
        super().__init__()
        sequence_dropout = sequence.pop("dropout", 0)
        self.sequence = SequenceRegistry.build(**sequence)
        self.sequence_dropout = nn.Dropout(sequence_dropout)
        self.config = self.sequence.config
        self.out_channels = self.config.hidden_size

    def forward(self, sequence: NestedTensor | Tensor, *args, **kwargs) -> tuple[FlatDict, FlatDict]:
        attentions = None
        input_ids, attention_mask = sequence.tensor, sequence.mask
        sequence_output = self.sequence(input_ids.int(), attention_mask)
        if "last_hidden_state" in sequence_output:
            sequence_output["last_hidden_state"] = self.sequence_dropout(sequence_output["last_hidden_state"])
        elif "logits" in sequence_output:
            sequence_output["last_hidden_state"] = self.sequence_dropout(sequence_output["logits"])
        else:
            raise ValueError("No token output")
        if "pooler_output" in sequence_output:
            sequence_output["pooler_output"] = self.sequence_dropout(sequence_output["pooler_output"])
        elif "logits" in sequence_output:
            sequence_output["pooler_output"] = self.sequence_dropout(
                sequence_output["logits"].mean(dim=1, keepdim=True)
            )
        else:
            raise ValueError("No sequence output")
        if "attentions" in sequence_output:
            attentions = torch.stack(sequence_output["attentions"], dim=1).detach()

        return sequence_output, attentions
