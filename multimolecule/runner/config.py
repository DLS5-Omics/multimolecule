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

import os
from pathlib import Path
from typing import Any, List, Optional

from chanfig import Config
from transformers import PretrainedConfig


class DataConfig(Config):
    root: str = "."
    train: Optional[str]
    validation: Optional[str]
    test: Optional[str]
    feature_cols: Optional[List] = None
    label_col: Optional[str] = None
    truncation: bool = True


class NetworkConfig(Config):
    backbone: Config
    heads: Config
    neck: Optional[Config] = None


class OptimConfig(Config):
    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    pretrained_ratio: float = 1e-2


class EmaConfig(Config):
    enabled: bool = False
    coerce_dtype: bool = True
    beta: float = 0.999
    update_after_step: int = 0
    update_every: int = 10


class MultiMoleculeConfig(Config):
    name: str
    seed: int = 1016

    balance: str = "ew"
    platform: str = "torch"
    training: bool = True

    pretrained: Optional[str]
    use_pretrained: bool = True
    transformers: PretrainedConfig
    epoch_end: int = 20

    data: DataConfig

    tensorboard: bool = True
    save_interval: int = 10

    art: bool = True
    allow_tf32: bool = True
    reduced_precision_reduction: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datas = Config(default_factory=DataConfig)
        self.dataloader.batch_size = 32
        self.network = NetworkConfig()
        self.optim = OptimConfig()
        self.ema = EmaConfig()
        self.sched.final_lr = 0

    def post(self):
        super().post()
        if "pretrained" not in self and "checkpoint" not in self:
            raise ValueError("Either one of `pretrained` or `checkpoint` must be specified")
        if "data" in self:
            if self.datas:
                raise ValueError("Only one of `data` or `datas` can be specified, but not both")
            del self.datas
        if "pretrained" in self:
            self["network.backbone.sequence.name"] = self.get("pretrained")
        self.name = str(self.name) if "name" in self else self.get_name(self.get("pretrained", "null"))
        self["network.backbone.sequence.use_pretrained"] = self.use_pretrained

    def get_name(self, pretrained: str) -> str:
        if os.path.exists(pretrained):
            path = Path(pretrained)
            if os.path.isfile(pretrained):
                pretrained = str(path.relative_to(path.parents[1]).with_suffix(""))
            else:
                pretrained = path.stem
        name = pretrained.replace("/", "--")
        if "optim" in self:
            optim_name = self.optim.get("name", "no")
            name += f"-{self.optim.lr}@{optim_name}"
        return name + f"-{self.seed}"

    def set(self, key: str, value: Any):
        if key == "data" and isinstance(value, str):
            value = DataConfig(root=value)
        elif key == "datas":
            for k, v in value.items():
                if isinstance(v, str):
                    value[k] = DataConfig(root=v)
        super().set(key, value)
