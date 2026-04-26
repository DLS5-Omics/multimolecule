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


import torch
from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn
from transformers import AutoConfig

from .registry import SEQUENCES


@SEQUENCES.register("onehot")
class OneHot(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(str(pretrained))
        self.module = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

    def forward(self, input_ids: NestedTensor | Tensor, attention_mask: Tensor | None = None) -> FlatDict:
        output = FlatDict()
        output["last_hidden_state"] = self.module(input_ids)
        if isinstance(output["last_hidden_state"], NestedTensor):
            output["pooler_output"] = torch.stack([hidden_state.sum(0) for hidden_state in output["last_hidden_state"]])
        else:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            valid_length = attention_mask.sum(dim=1)
            output["pooler_output"] = torch.stack(
                [t[: valid_length[i]].sum(0) for i, t in enumerate(output["last_hidden_state"])]
            )
        return output
