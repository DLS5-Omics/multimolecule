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
from collections.abc import Mapping
from functools import cached_property, partial
from itertools import chain
from pathlib import Path
from typing import Any, cast
from warnings import warn

import danling as dl
import torch
import torch.distributed as dist
from chanfig import FlatDict, NestedDict
from danling import METRICS, MultiTaskMetrics, NestedTensor
from danling.runners.utils import RunnerMode
from datasets import get_dataset_split_names
from datasets.data_files import DataFilesDict, get_data_patterns
from lazy_imports import try_import
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from multimolecule import defaults
from multimolecule.data import DATASETS, Dataset, no_collate
from multimolecule.modules import MODELS, HeadConfig, ModelBase
from multimolecule.tasks import Task

from .config import Config
from .registry import RUNNERS

with try_import() as ema_import:
    from ema_pytorch import EMA


@RUNNERS.register("multimolecule", default=True)
class Runner(dl.Runner):
    config: Config
    model: ModelBase
    optimizer: Any

    def __init__(self, config: Config | Mapping[str, Any]) -> None:
        if not isinstance(config, Config):
            config = Config(config)
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        self.datasets = self.build_datasets()
        self.model = cast(ModelBase, MODELS.build(**self.network))
        self._build_ema()
        self.train_metrics = self.build_metrics(mode="stream")
        self.evaluate_metrics = self.build_metrics(mode="global")

    def __post_init__(self) -> None:
        super().__post_init__()
        self.config.yaml(os.path.join(self.workspace.dir, "trainer.yaml"))

    @cached_property
    def pretrained_name(self) -> str:
        sequence_config = self.config.network.backbone.get("sequence", {})
        pretrained = self.config.pretrained or sequence_config.get("name")
        if pretrained is None:
            raise ValueError("Either `pretrained` or `network.backbone.sequence.name` must be specified")
        return pretrained

    @cached_property
    def tasks(self) -> NestedDict[str, Task]:
        if not self.datasets:
            raise ValueError("No datasets found")
        dataset = self.datasets.train if "train" in self.datasets else next(iter(self.datasets.values()))
        return dataset.tasks

    @cached_property
    def task(self) -> Task:
        if len(self.tasks) != 1:
            raise ValueError(f"Expected exactly one task, got {len(self.tasks)}")
        return next(iter(self.tasks.values()))

    @cached_property
    def network(self) -> NestedDict:
        network = NestedDict(self.config.network)
        sequence_config = network.setdefault("backbone", NestedDict()).setdefault("sequence", NestedDict())
        if self.config.pretrained is not None and "name" not in sequence_config:
            sequence_config.name = self.config.pretrained
        if "use_pretrained" not in sequence_config:
            sequence_config.use_pretrained = self.config.use_pretrained
        heads = network.setdefault("heads", NestedDict())
        legacy_head = network.pop("head", None)
        for name, task in self.tasks.items():
            head = HeadConfig(num_labels=task.num_labels, problem_type=task.type, type=task.level)
            if name in heads:
                heads[name].merge(head, overwrite=False)
            elif legacy_head is not None and len(self.tasks) == 1:
                heads[name] = NestedDict(legacy_head)
                heads[name].merge(head, overwrite=False)
            else:
                heads[name] = NestedDict(head)
        return network

    def _build_ema(self) -> None:
        ema_config = self.config.get("ema")
        if not isinstance(ema_config, Mapping):
            return
        ema_kwargs = NestedDict(ema_config)
        enabled = bool(ema_kwargs.pop("enabled", False))
        if not enabled:
            return
        ema_import.check()
        self.ema = EMA(self.model, include_online_model=False, **ema_kwargs)

    def build_optimizer(self) -> None:
        if getattr(self, "optimizer", None) is not None or self.model is None:
            return
        optim_config = self.config.get("optim") or self.config.get("optimizer")
        if not isinstance(optim_config, Mapping) or not optim_config:
            return
        optim_kwargs = NestedDict(optim_config)
        pretrained_ratio = optim_kwargs.pop("pretrained_ratio", None)
        model = self.unwrap(self.model)
        if (
            pretrained_ratio is not None
            and isinstance(model, ModelBase)  # noqa: W503
            and "lr" in optim_kwargs  # noqa: W503
            and "weight_decay" in optim_kwargs  # noqa: W503
        ):
            parameters = model.trainable_parameters(
                lr=optim_kwargs.lr,
                weight_decay=optim_kwargs.weight_decay,
                pretrained_ratio=pretrained_ratio,
            )
        else:
            parameters = list(self.iter_optimizer_parameters())
        self.optimizer = dl.OPTIMIZERS.build(params=parameters, **optim_kwargs)

    def _resolve_auto_restore_target(self) -> tuple[str, Mapping[Any, Any] | os.PathLike | str | bytes] | None:
        resume_source = self.config.get("resume")
        if resume_source:
            return ("checkpoint", resume_source)

        checkpoint_source = self.config.get("checkpoint_path")
        if checkpoint_source:
            return ("checkpoint", checkpoint_source)

        legacy_checkpoint = self.config.get("checkpoint")
        if legacy_checkpoint and not isinstance(legacy_checkpoint, Mapping):
            return ("checkpoint", legacy_checkpoint)

        if self.config.get("auto_resume", False):
            return ("checkpoint", self._auto_resume_source())

        pretrained_checkpoint = self.config.get("model_pretrained") or self.config.get("load_pretrained")
        if pretrained_checkpoint:
            return ("pretrained", pretrained_checkpoint)
        return None

    def train_step(self, data: Mapping[str, Any]) -> tuple[Any, Tensor | None]:
        data = self.to_device(data)
        with self.train_context():
            output = self.model(**data)
            loss = self._output_loss(output)
            self._update_metrics(output, data)
            self.backward(loss)
            self.step()
        return output, loss

    def evaluate_step(self, data: Mapping[str, Any]) -> tuple[Any, Tensor | None]:
        data = self.to_device(data)
        model = self.ema or self.model
        with self.forward_context():
            output = model(**data)
            loss = self._output_loss(output, required=False)
        self._update_metrics(output, data)
        return output, loss

    @torch.inference_mode()
    def infer(self, split: str = "infer") -> NestedDict | FlatDict | list:
        self.mode = RunnerMode.infer
        self.split = split
        loader = self.dataloaders[split]
        model = self.ema or self.model
        model.eval()
        preds: dict[str, list] = {name: [] for name in self.tasks}
        labels: dict[str, list] = {}

        for data in tqdm(loader, total=len(loader), disable=self.distributed and not self.is_main_process):
            data = self.to_device(data)
            with self.forward_context():
                output = model(**data)
            for task, task_output in output.items():
                preds[task].extend(self._as_list(task_output.logits))
                if task in data:
                    labels.setdefault(task, []).extend(self._as_list(data[task]))

        if self.distributed and dist.is_available() and dist.is_initialized():
            preds = {task: self._gather_list(values) for task, values in preds.items()}
            labels = {task: self._gather_list(values) for task, values in labels.items()}

        if labels:
            if len(preds) == 1:
                task = next(iter(preds))
                return FlatDict(predict=preds[task], label=labels.get(task, []))
            return NestedDict(
                {task: {"predict": values, "label": labels.get(task, [])} for task, values in preds.items()}
            )
        if len(preds) == 1:
            return next(iter(preds.values()))
        return FlatDict(preds)

    def _output_loss(self, output: Mapping[str, Any], required: bool = True) -> Tensor | None:
        losses = [task_output.loss for task_output in output.values() if task_output.loss is not None]
        if not losses:
            if required:
                raise ValueError("Model output does not contain a loss. Did you pass labels in the batch?")
            return None
        return sum(losses)

    def _update_metrics(self, output: Mapping[str, Any], data: Mapping[str, Any]) -> None:
        if self.metrics is None:
            return
        payload = {
            task: {"input": task_output.logits, "target": data[task]}
            for task, task_output in output.items()
            if task in data
        }
        if not payload:
            return
        if isinstance(self.metrics, MultiTaskMetrics):
            self.metrics.update(payload)
            return
        task_payload = next(iter(payload.values()))
        self.metrics.update(task_payload["input"], task_payload["target"])

    def build_metrics(self, mode: str):
        if len(self.tasks) == 1:
            task = self.task
            return METRICS.build(
                type=task.type,
                mode=mode,
                num_labels=task.num_labels,
                distributed=self.distributed,
                device=self.device,
            )
        metrics = MultiTaskMetrics(aggregate="macro")
        for name, task in self.tasks.items():
            metrics[name] = METRICS.build(
                type=task.type,
                mode=mode,
                num_labels=task.num_labels,
                distributed=self.distributed,
                device=self.device,
            )
        return metrics

    def build_datasets(self) -> NestedDict[str, Dataset]:
        data_config = self.config.data
        if isinstance(data_config, str):
            data_config = NestedDict(root=data_config)
        return self._build_dataset(NestedDict(data_config))

    def _build_dataset(self, config: NestedDict, name: str | None = None) -> NestedDict:
        root = config.pop("root", None)
        if root is None:
            raise ValueError(f"Unable to build dataset for {config}, root is not specified.")
        local_root = Path(root).expanduser().resolve()
        if name is None:
            name = "/".join(local_root.parts[-2:])
        ratio = config.pop("ratio", None)
        try:
            is_hf_dataset = bool(get_dataset_split_names(root))
        except FileNotFoundError:
            is_hf_dataset = False
        if local_root.is_dir():
            return self._build_local_dataset(config, str(local_root), ratio, name)
        if is_hf_dataset:
            return self._build_hf_dataset(config, root, ratio, name)
        raise ValueError(
            f"Dataset root '{root}' is invalid. It must be either a Hugging Face dataset ID "
            "or a path to an existing local directory."
        )

    def _build_local_dataset(self, config: NestedDict, root: str, ratio: float | int | None, name: str) -> NestedDict:
        train_splits, other_splits = self._configured_splits(config)
        splits = train_splits + other_splits
        if not splits:
            for split, data_files in DataFilesDict.from_local_or_remote(get_data_patterns(root), root).items():
                split = str(split)
                if len(data_files) > 1:
                    for idx, data_file in enumerate(data_files):
                        split_name = f"{split}-{idx:05d}-of-{len(data_files):05d}"
                        config[split_name] = data_file
                        if split in defaults.TRAIN_SPLITS:
                            train_splits.append(split_name)
                        else:
                            other_splits.append(split_name)
                else:
                    config[split] = data_files[0]
                    if split in defaults.TRAIN_SPLITS:
                        train_splits.append(split)
                    else:
                        other_splits.append(split)
            splits = train_splits + other_splits
        if not splits:
            raise ValueError(f"No splits found for dataset {name}. Please specify at least one split in the config.")

        print(f"Building local dataset {name}")
        dataset_factory = self._dataset_factory(config, splits)
        dataset = NestedDict()
        if self.config.training:
            for split in train_splits:
                dataset[split] = dataset_factory(
                    os.path.join(root, config[split]), split=split, train=True, ratio=ratio
                )
        elif train_splits:
            warn("Training is disabled, ignoring training splits", RuntimeWarning, stacklevel=2)
        for split in other_splits:
            dataset[split] = dataset_factory(os.path.join(root, config[split]), split=split, train=False)
        return dataset

    def _build_hf_dataset(self, config: NestedDict, root: str, ratio: float | int | None, name: str) -> NestedDict:
        splits = [k for k in defaults.DATASET_SPLITS if config.get(k) is not None] or get_dataset_split_names(root)
        train_splits = [split for split in splits if split in defaults.TRAIN_SPLITS]
        other_splits = [split for split in splits if split not in train_splits]
        print(f"Building Hugging Face dataset {name}")
        dataset_factory = self._dataset_factory(config, splits)
        dataset = NestedDict()
        if self.config.training:
            for split in train_splits:
                dataset[split] = dataset_factory(root, split=split, train=True, ratio=ratio)
        elif train_splits:
            warn("Training is disabled, ignoring training splits", RuntimeWarning, stacklevel=2)
        for split in other_splits:
            dataset[split] = dataset_factory(root, split=split, train=False)
        return dataset

    def _dataset_factory(self, config: NestedDict, ignored_keys: list[str]):
        kwargs = NestedDict({k: v for k, v in config.items() if k not in ignored_keys and v is not None})
        if "label_col" in kwargs and "label_cols" not in kwargs:
            kwargs.label_cols = [kwargs.pop("label_col")]
        return partial(DATASETS.build, tokenizer=self.tokenizer, auto_rename_label_col=True, **kwargs)

    @staticmethod
    def _configured_splits(config: NestedDict) -> tuple[list[str], list[str]]:
        splits = [k for k in defaults.DATASET_SPLITS if config.get(k) is not None]
        train_splits = [split for split in splits if split in defaults.TRAIN_SPLITS]
        other_splits = [split for split in splits if split not in train_splits]
        return train_splits, other_splits

    @staticmethod
    def _as_list(value: Any) -> list:
        if isinstance(value, NestedTensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Tensor):
            value = value.detach().cpu()
            if value.ndim > 0 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            return value.tolist()
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _gather_list(values: list) -> list:
        gathered: list[list[Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, values)
        return list(chain.from_iterable(values or [] for values in gathered))

    @staticmethod
    def collate_fn(batch: Any) -> Any:
        return no_collate(batch)
