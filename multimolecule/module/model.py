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

from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn

from .backbones import BackboneRegistry
from .heads import HeadRegistry
from .necks import NeckRegistry
from .registry import ModelRegistry


@ModelRegistry.register(default=True)
class MultiMoleculeModel(nn.Module):
    def __init__(
        self,
        backbone: dict,
        heads: dict,
        neck: dict | None = None,
        max_length: int = 1024,
        truncation: bool = False,
    ):
        super().__init__()

        # Backbone
        self.backbone = BackboneRegistry.build(**backbone)
        backbone = self.backbone.config
        out_channels = self.backbone.out_channels

        # Neck
        if neck:
            num_discrete = self.backbone.num_discrete
            num_continuous = self.backbone.num_continuous
            embed_dim = self.backbone.sequence.config.hidden_size
            attention_heads = self.backbone.sequence.config.num_attention_heads
            neck.update(
                {
                    "num_discrete": num_discrete,
                    "num_continuous": num_continuous,
                    "embed_dim": embed_dim,
                    "attention_heads": attention_heads,
                    "max_length": max_length,
                    "truncation": truncation,
                }
            )
            self.neck = NeckRegistry.build(**neck)
            out_channels = self.neck.out_channels
        else:
            self.neck = None

        # Heads
        for head in heads.values():
            if "hidden_size" not in head or head["hidden_size"] is None:
                head["hidden_size"] = out_channels
        self.heads = nn.ModuleDict({name: HeadRegistry.build(backbone, head) for name, head in heads.items()})
        if any(getattr(h, "requires_attention", False) for h in self.heads.values()):
            self.backbone.sequence.config.output_attentions = True

    def forward(
        self,
        sequence: NestedTensor | Tensor,
        discrete: Tensor | None = None,
        continuous: Tensor | None = None,
        dataset: str | None = None,
        **labels: NestedTensor | Tensor,
    ) -> FlatDict:
        ret = FlatDict()
        output, _ = self.backbone(sequence, discrete, continuous)
        if self.neck is not None:
            output = self.neck(**output)
        for task, label in labels.items():
            ret[task] = self.heads[task](output, input_ids=sequence, labels=label)
        return ret
