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

from typing import List

from chanfig import Config
from transformers import PretrainedConfig


class DataConfig(Config):
    root: str = "."
    train: str | None
    validation: str | None
    test: str | None
    feature_cols: List | None = None
    label_cols: List | None = None
    truncation: bool = True


class OptimConfig(Config):
    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-2


class MultiMoleculeConfig(Config):

    platform: str = "torch"
    training: bool = True

    pretrained: str
    use_pretrained: bool = True
    transformers: PretrainedConfig
    epoch_end: int = 20

    data: DataConfig

    tensorboard: bool = True
    save_interval: int = 10
    seed: int = 1013
    art: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datas = Config(default_factory=DataConfig)
        self.dataloader.batch_size = 32
        self.optim = OptimConfig()
        self.sched.final_lr = 0

    def post(self):
        if "data" in self:
            if self.datas:
                raise ValueError("Only one of `data` or `datas` can be specified, but not both")
            del self.datas
        self["network.backbone.sequence.name"] = self.pretrained
        self["network.backbone.sequence.use_pretrained"] = self.use_pretrained
        self.name = f"{self.pretrained}-{self.optim.lr}@{self.optim.name}-{self.seed}"
