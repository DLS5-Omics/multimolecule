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

import torch
from chanfig import FlatDict
from danling.modules import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor, nn

from .registry import NeckRegistry

MAX_LENGTH = 1024


@NeckRegistry.register("bert")
class BERTNeck(nn.Module):
    def __init__(  # pylint: disable=keyword-arg-before-vararg
        self,
        num_discrete: int,
        num_continuous: int,
        embed_dim: int,
        attention_heads: int,
        num_layers: int = 6,
        max_length: int | None = None,
        truncation: bool = False,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cls_token_dis = nn.Parameter(torch.zeros(embed_dim))
        self.cls_token_con = nn.Parameter(torch.zeros(embed_dim))
        if max_length is None:
            if truncation:
                max_length = MAX_LENGTH + 1 + num_discrete + 1 + num_continuous
            else:
                max_length = MAX_LENGTH * 4 + 1 + num_discrete + 1 + num_continuous
        self.max_length = max_length
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_length, embed_dim))
        bert_layer = TransformerEncoderLayer(
            embed_dim, attention_heads, *args, dropout=dropout, attn_dropout=dropout, ffn_dropout=dropout, **kwargs
        )
        self.bert = TransformerEncoder(bert_layer, num_layers)
        self.out_channels = embed_dim
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token_dis, std=0.2)
        nn.init.trunc_normal_(self.cls_token_con, std=0.2)

    def forward(
        self,
        cls_token: Tensor | None = None,
        all_tokens: Tensor | None = None,
        discrete: Tensor | None = None,
        continuous: Tensor | None = None,
    ) -> FlatDict:
        ret = FlatDict()
        if cls_token is not None:
            ret["cls_token"] = self._forward(cls_token, discrete, continuous)
        if all_tokens is not None:
            ret["all_tokens"] = self._forward(all_tokens, discrete, continuous)
        return ret

    def _forward(
        self,
        sequence: Tensor,
        discrete: Tensor | None = None,
        continuous: Tensor | None = None,
    ) -> Tensor:
        if sequence is None:
            raise ValueError("sequence should not be None.")
        if sequence.dim() == 2:
            sequence = sequence[:, None]
        batch_size, seq_len, _ = sequence.shape
        output = sequence
        if discrete is not None:
            cls_token_dis = self.cls_token_dis.expand(batch_size, 1, -1)
            output = torch.cat((output, cls_token_dis, discrete), dim=1)
        if continuous is not None:
            cls_token_con = self.cls_token_con.expand(batch_size, -1)[:, None]
            output = torch.cat((output, cls_token_con, continuous), dim=1)
        all_len = output.shape[1]
        if all_len > self.pos_embed.shape[1]:
            raise ValueError("sequence length is out of range.")
        output = output + self.pos_embed[:, 0:all_len, :]
        output = self.bert(output)[0][:, 0:seq_len, :]
        if seq_len == 1:
            output = output.squeeze(1)
        return output
