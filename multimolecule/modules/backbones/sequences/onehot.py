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

from typing import Tuple

import torch
from chanfig import FlatDict
from torch import Tensor, nn
from transformers import AutoConfig

from .registry import SEQUENCES


@SEQUENCES.register("onehot")
class OneHot(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(str(pretrained))
        self.module = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> FlatDict | Tuple[Tensor, Tensor]:
        last_hidden_state = self.module(input_ids)
        valid_length = attn_mask.sum(dim=1)
        pooler_output = torch.stack([t[: valid_length[i]].sum(0) for i, t in enumerate(last_hidden_state)])
        return FlatDict(last_hidden_state=last_hidden_state, pooler_output=pooler_output)
