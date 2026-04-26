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
from typing import Any

import chanfig
import danling as dl

DEFAULT_TRAIN_EPOCHS: int = 20


class DataConfig(chanfig.Config):
    root: str = "."
    train: str | None = None
    validation: str | None = None
    valid: str | None = None
    val: str | None = None
    test: str | None = None
    infer: str | None = None
    inference: str | None = None
    feature_cols: list[str] | None = None
    label_cols: list[str] | None = None
    label_col: str | None = None
    sequence_cols: list[str] | None = None
    ignored_cols: list[str] | None = None
    truncation: bool = True
    max_seq_length: int | None = None
    ratio: float | int | None = None


class DataloaderConfig(chanfig.Config):
    batch_size: int = 32
    num_workers: int = 4


class NetworkConfig(chanfig.Config):
    backbone: chanfig.NestedDict
    heads: chanfig.NestedDict
    neck: chanfig.NestedDict | None = None

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if "backbone" not in self:
            self.backbone = chanfig.NestedDict(sequence=chanfig.NestedDict())
        if "heads" not in self:
            self.heads = chanfig.NestedDict()


class OptimConfig(chanfig.Config):
    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    pretrained_ratio: float | None = None


class SchedulerConfig(chanfig.Config):
    type: str = "cosine"
    final_lr: float = 0.0


class EmaConfig(chanfig.Config):
    enabled: bool = False
    coerce_dtype: bool = True
    beta: float = 0.9999
    update_after_step: int = 0
    update_every: int = 8
    update_model_with_ema_every: int | None = None
    update_model_with_ema_beta: float = 0.0


class Config(dl.RunnerConfig):
    seed: int | None = 1016
    training: bool = True

    runner: str = "multimolecule"
    platform: str | None = None

    pretrained: str | None = None
    use_pretrained: bool = True

    steps: int | None = None
    epochs: int | None = None

    data: DataConfig | str
    dataloader: DataloaderConfig
    network: NetworkConfig
    optim: OptimConfig
    sched: SchedulerConfig
    ema: EmaConfig

    allow_tf32: bool = True
    reduced_precision_reduction: bool = False

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if "dataloader" not in self:
            self.dataloader = DataloaderConfig()
        elif not isinstance(self.dataloader, DataloaderConfig):
            self.dataloader = DataloaderConfig(self.dataloader)
        if "network" not in self:
            self.network = NetworkConfig()
        elif not isinstance(self.network, NetworkConfig):
            self.network = NetworkConfig(self.network)
        if "optim" not in self:
            self.optim = OptimConfig()
        elif not isinstance(self.optim, OptimConfig):
            self.optim = OptimConfig(self.optim)
        if "sched" not in self:
            self.sched = SchedulerConfig()
        elif not isinstance(self.sched, SchedulerConfig):
            self.sched = SchedulerConfig(self.sched)
        if "ema" not in self:
            self.ema = EmaConfig()
        elif not isinstance(self.ema, EmaConfig):
            self.ema = EmaConfig(self.ema)
        if self.platform is not None and "stack" not in self:
            self.stack = self.platform

    def post(self) -> None:
        super().post()
        if self.epochs is None and self.steps is None:
            self.epochs = DEFAULT_TRAIN_EPOCHS

        sequence_config = self.network.backbone.setdefault("sequence", chanfig.NestedDict())
        if self.pretrained is not None and "name" not in sequence_config:
            sequence_config.name = self.pretrained
        if "use_pretrained" not in sequence_config:
            sequence_config.use_pretrained = self.use_pretrained
        if self.pretrained is None and "name" not in sequence_config:
            raise ValueError("Either `pretrained` or `network.backbone.sequence.name` must be specified")
        if "data" not in self:
            raise ValueError("`data` must be specified")
        if "name" not in self:
            self.name = self.get_name(self.pretrained or sequence_config.name)

    def get_name(self, pretrained: str) -> str:
        if os.path.exists(pretrained):
            path = Path(pretrained)
            if path.is_file():
                pretrained = str(path.relative_to(path.parents[1]).with_suffix(""))
            else:
                pretrained = path.stem
        name = pretrained.replace("/", "--")
        if self.get("optim"):
            name += f"-{self.optim.lr}@{self.optim.get('type', 'no')}"
        return f"{name}-{self.seed}"

    def set(self, key: str, value: Any) -> None:
        if key == "data" and isinstance(value, str):
            value = DataConfig(root=value)
        super().set(key, value)


__all__ = [
    "Config",
    "DataConfig",
    "DataloaderConfig",
    "EmaConfig",
    "NetworkConfig",
    "OptimConfig",
    "SchedulerConfig",
]
