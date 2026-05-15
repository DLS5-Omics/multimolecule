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
    r"""
    Dataset configuration for the runner.

    `data` accepts either a string (treated as `root`) or a mapping (parsed into this class). The runner resolves
    `root` to either a local directory or a Hugging Face dataset ID and uses the split keys below to locate files;
    when none are given for a local dataset, splits are discovered with Hugging Face's standard data-file patterns.

    Args:
        root:
            Dataset root. Either a local directory containing split files or a Hugging Face Hub dataset ID such as
            `multimolecule/rnacentral`.
        train:
            Training split file (local) or split name (Hugging Face).
        validation:
            Validation split file (local) or split name (Hugging Face).

            `valid` and `val` are accepted as aliases for compatibility with third-party configs.
        valid:
            Alias for `validation`.
        val:
            Alias for `validation`.
        test:
            Test split file (local) or split name (Hugging Face).
        infer:
            Inference split file (local) or split name (Hugging Face).
        inference:
            Alias for `infer`.
        sequence_cols:
            Columns to treat as biological sequences. Forwarded to [`Dataset`][multimolecule.data.Dataset] for
            tokenization.
        feature_cols:
            Non-sequence input columns retained alongside `label_cols`.
        label_cols:
            Label columns. Task metadata (level / type / num_labels) is inferred per column and one head is built
            per label.
        label_col:
            Single-label shortcut; promoted to `[label_col]` when `label_cols` is unset.
        ignored_cols:
            Columns to drop before training.
        truncation:
            Whether to truncate sequences longer than `max_seq_length`.
        max_seq_length:
            Maximum sequence length in tokens.
        ratio:
            Optional sub-sampling fraction (float in `(0, 1]`) or row count (int) applied to training splits only.
            Useful for smoke tests.
    """

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
    r"""
    DataLoader configuration.

    Additional keys are forwarded to [`torch.utils.data.DataLoader`][torch.utils.data.DataLoader] through the
    underlying DanLing dataloader builder.

    Args:
        batch_size:
            Per-process batch size.
        num_workers:
            Number of worker processes used to load batches.
    """

    batch_size: int = 32
    num_workers: int = 4


class NetworkConfig(chanfig.Config):
    r"""
    Model configuration consumed by [`MODELS.build`][multimolecule.modules.MODELS].

    `network.backbone.sequence` is the only required sub-tree; the runner populates `backbone.sequence.name` and
    `backbone.sequence.use_pretrained` from top-level `pretrained` / `use_pretrained` when those keys are not already
    set. One head is added to `network.heads` for each task inferred from the dataset labels, with user-provided
    head settings (e.g. `dropout`, `hidden_size`) preserved through merge-without-overwrite.

    Args:
        backbone:
            Backbone configuration. Must contain a `sequence` sub-dict whose `name` resolves to a Hugging Face
            model identifier or a local path loadable as a [`MultiMoleculeModel`][multimolecule.MultiMoleculeModel].
        heads:
            Per-task head configuration. Each entry is merged with the task metadata
            (`num_labels` / `problem_type` / `type`) inferred from `data.label_cols`.
        neck:
            Optional neck applied between backbone and heads.
    """

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
    r"""
    Optimizer configuration.

    Forwarded to [`dl.OPTIMIZERS.build`][danling.OPTIMIZERS] after popping `pretrained_ratio`.

    Args:
        type:
            Optimizer name registered in [`dl.OPTIMIZERS`][danling.OPTIMIZERS].
        lr:
            Base learning rate applied to newly initialised parameters (heads, necks, ...).
        weight_decay:
            Base weight decay applied to newly initialised parameters.
        pretrained_ratio:
            Multiplier applied to `lr` and `weight_decay` for parameters belonging to the pretrained backbone.
            Useful for fine-tuning a backbone alongside freshly initialised task heads. Both `lr` and `weight_decay`
            must be set for this to take effect.
    """

    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    pretrained_ratio: float | None = None


class SchedulerConfig(chanfig.Config):
    r"""
    Learning rate scheduler configuration.

    Forwarded to DanLing's scheduler builder. Common warmup keys (`warmup_ratio`, `warmup_steps`, ...) are accepted
    and passed through unchanged.

    Args:
        type:
            Scheduler name.
        final_lr:
            Target learning rate at the end of training.
    """

    type: str = "cosine"
    final_lr: float = 0.0


class EmaConfig(chanfig.Config):
    r"""
    Exponential moving average configuration.

    When `enabled`, the runner instantiates an `ema_pytorch.EMA` wrapper around the trained model and uses it for
    evaluation and inference. Remaining fields are forwarded to `EMA` unchanged.

    Args:
        enabled:
            Whether EMA is active.
        coerce_dtype:
            Coerce EMA weights to the online model's dtype.
        beta:
            EMA decay.
        update_after_step:
            Skip EMA updates until this many optimizer steps have elapsed.
        update_every:
            Run an EMA update once every N optimizer steps.
        update_model_with_ema_every:
            If set, periodically copy EMA weights back onto the online model.
        update_model_with_ema_beta:
            Mixing factor for the periodic EMA-to-online copy.
    """

    enabled: bool = False
    coerce_dtype: bool = True
    beta: float = 0.9999
    update_after_step: int = 0
    update_every: int = 8
    update_model_with_ema_every: int | None = None
    update_model_with_ema_beta: float = 0.0


class Config(dl.RunnerConfig):
    r"""
    Top-level runner configuration.

    Extends [`dl.RunnerConfig`][danling.runners.RunnerConfig] with MultiMolecule defaults and validation. The runner
    accepts either a fully-constructed `Config` instance or any mapping that this class can be built from.

    The `name` attribute is auto-derived in [`post`][multimolecule.runner.Config.post] from the pretrained
    identifier, optimizer settings, and seed when the user does not set it explicitly.

    Args:
        seed:
            Base random seed.
        training:
            When `False`, training splits are ignored and the runner is usable for evaluation/inference. Set
            automatically by the `mmtrain` / `mmevaluate` / `mminfer` entry points in `multimolecule.apis.run`.
        runner:
            Registry key resolved through [`RUNNERS`][multimolecule.runner.registry.RUNNERS].
        platform:
            Alias for `stack`. When set, the value is copied into `self.stack` during `__post_init__`. Accepts the
            same values as DanLing's stack selector (`ddp`, `torch`, `parallel`, `deepspeed`, ...).
        pretrained:
            Pretrained backbone identifier (Hugging Face Hub repo or local path). Copied into
            `network.backbone.sequence.name` when that key is not already set.
        use_pretrained:
            When `False`, build the architecture from the pretrained config but reinitialise weights from scratch.
        steps:
            Training step budget. Mutually exclusive with `epochs`.
        epochs:
            Training epoch budget. Defaults to
            [`DEFAULT_TRAIN_EPOCHS`][multimolecule.runner.config.DEFAULT_TRAIN_EPOCHS] when both `steps` and
            `epochs` are unset.
        data:
            Dataset configuration. A bare string is promoted to `DataConfig(root=<string>)`.
        dataloader:
            DataLoader configuration.
        network:
            Model configuration.
        optim:
            Optimizer configuration.
        sched:
            Learning rate scheduler configuration.
        ema:
            Optional EMA configuration.
        allow_tf32:
            Whether to allow TF32 matmul / cuDNN kernels on Ampere+ GPUs.
        reduced_precision_reduction:
            Whether to allow reduced-precision reductions for fp16 / bf16 matmul accumulators.
    """

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
