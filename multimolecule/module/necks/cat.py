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
from torch import Tensor

from .registry import NeckRegistry


@NeckRegistry.register("cat")
class CatNeck:  # pylint: disable=too-few-public-methods
    def __init__(self, embed_dim: int):
        self.out_channels = embed_dim * 2

    def __call__(
        self,
        cls_token: Tensor | None = None,
        all_tokens: Tensor | None = None,
        discrete: Tensor | None = None,
        continuous: Tensor | None = None,
    ) -> FlatDict:
        ret = FlatDict()
        if cls_token is not None:
            ret.cls_token = torch.cat(tuple(i for i in (cls_token, discrete, continuous) if i is not None), -1)
        if all_tokens is not None:
            ret.all_tokens = torch.cat(tuple(i for i in (all_tokens, discrete, continuous) if i is not None), -1)
        return ret
