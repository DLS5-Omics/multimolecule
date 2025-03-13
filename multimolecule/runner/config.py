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
from typing import Any, List, Optional, Union

import chanfig
from transformers import PretrainedConfig


class DataConfig(chanfig.Config):
    root: str = "."
    train: Optional[str]
    validation: Optional[str]
    test: Optional[str]
    feature_cols: Optional[List] = None
    label_col: Optional[str] = None
    truncation: bool = True


class DataloaderConfig(chanfig.Config):
    batch_size: int = 32
    num_workers: int = 4


class NetworkConfig(chanfig.Config):
    backbone: chanfig.Config
    heads: chanfig.Config
    neck: Optional[chanfig.Config] = None


class OptimConfig(chanfig.Config):
    type: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    pretrained_ratio: Optional[float] = None


class SchedulerConfig(chanfig.Config):
    type: str = "cosine"
    final_lr: float = 0


class EmaConfig(chanfig.Config):
    enabled: bool = False
    coerce_dtype: bool = True
    beta: float = 0.9999
    update_after_step: int = 0
    update_every: int = 8
    update_model_with_ema_every: Optional[int] = None
    update_model_with_ema_beta: float = 0


class Config(chanfig.Config):
    name: str
    seed: int = 1016
    training: bool = True

    runner: str
    platform: str = "auto"

    pretrained: Optional[str]
    use_pretrained: bool = True
    transformers: PretrainedConfig
    epoch_end: int = 20

    data: Union[DataConfig, str]
    dataloader: DataloaderConfig
    network: NetworkConfig
    optim: OptimConfig
    sched: SchedulerConfig
    ema: EmaConfig

    tensorboard: bool = True
    save_interval: int = 10

    allow_tf32: bool = True
    reduced_precision_reduction: bool = False

    def __init__(self, *args, **kwargs):
        self.dataloader = DataloaderConfig()
        self.network = NetworkConfig()
        self.optim = OptimConfig()
        self.sched = SchedulerConfig()
        self.ema = EmaConfig()
        super().__init__(*args, **kwargs)

    def post(self):
        super().post()
        if "pretrained" not in self and "checkpoint" not in self:
            raise ValueError("Either one of `pretrained` or `checkpoint` must be specified")
        if "data" not in self:
            raise ValueError("`data` must be specified")
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
        if self.get("optim"):
            optim_name = self.optim.get("type", "no")
            name += f"-{self.optim.lr}@{optim_name}"
        return name + f"-{self.seed}"

    def set(self, key: str, value: Any):
        if key == "data" and isinstance(value, str):
            value = DataConfig(root=value)
        super().set(key, value)
