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

import math
import os
from functools import cached_property, partial
from typing import Any, Tuple
from warnings import warn

import danling as dl
import torch
from art import text2art
from chanfig import NestedDict
from danling import MultiTaskMetrics
from datasets import disable_progress_bars, get_dataset_split_names
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from transformers import AutoTokenizer

from multimolecule import defaults
from multimolecule.data import Dataset, DistributedMultiTaskSampler, MultiTaskDataset, MultiTaskSampler
from multimolecule.module import HeadConfig, ModelRegistry

from .config import MultiMoleculeConfig
from .metrics import MetricRegistry

disable_progress_bars()


class BaseRunner(dl.BaseRunner):

    all_datasets: NestedDict

    def __init__(self, config: MultiMoleculeConfig):
        if config.art:
            print(text2art("MultiMolecule", "rand-large"))
        super().__init__(config)
        self.name = config.name
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained)
        self.build_datasets()
        self.build_dataloaders()
        self.model = ModelRegistry.build(**self.network)
        if self.config.get("checkpoint"):
            ckpt = dl.load(self.config.checkpoint)
            model = ckpt.get("model", ckpt)
            parameters = self.model.load_state_dict(model, strict=False)
            if parameters.missing_keys:
                raise ValueError(f"Missing keys in model: {parameters.missing_keys}")
            if parameters.unexpected_keys:
                warn(f"Unexpected keys in model: {parameters.unexpected_keys}")
        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, find_unused_parameters=True, bucket_cap_mb=32, gradient_as_bucket_view=True
            )
        if self.config.training:
            if self.config.optim and not (self.config.platform == "deepspeed" and self.config.deepspeed.optimizer):
                self.optimizer = getattr(optim, self.config.optim.pop("name"))(
                    params=self.model.parameters(), **self.config.optim
                )
            if self.config.sched and not (self.config.platform == "deepspeed" and self.config.deepspeed.scheduler):
                self.scheduler = dl.optim.LRScheduler(self.optimizer, total_steps=self.total_steps, **self.config.sched)
        self.metrics = self.build_metrics()

    def __post_init__(self):
        super().__post_init__()
        self.yaml(os.path.join(self.dir, "trainer.yaml"))
        print(self)
        print(self.get_dataset_lengths())

    def train_step(self, data) -> Tuple[Any, torch.Tensor]:
        with self.autocast(), self.accumulate():
            pred = self.model(**data)
            loss = self.loss_fn(pred, data)
            self.advance(loss)
            self.metric_fn(pred, data)
        return pred, loss

    def evaluate_step(self, data) -> Tuple[Any, torch.Tensor]:
        pred = self.model(**data)
        loss = self.loss_fn(pred, data)
        self.metric_fn(pred, data)
        return pred, loss

    def loss_fn(self, pred, data):
        if self.balance == "rlw":
            loss = torch.stack([p["loss"] for p in pred.values()])
            weight = F.softmax(torch.randn(len(pred), device=loss.device, dtype=loss.dtype), dim=-1)
            return loss.T @ weight
        if self.balance == "gls":
            return math.prod(p["loss"] for p in pred.values()) ** (1 / len(pred))
        if self.balance != "ew":
            warn(f"Unknown balance method {self.balance}, using equal weighting.")
        return sum(p["loss"] for p in pred.values()) / len(pred)

    def metric_fn(self, pred, data):
        metric = self.metrics[data["dataset"]] if "dataset" in data else self.metrics
        metric.update({t: (p["logits"], data[t]) for t, p in pred.items()})

    @cached_property
    def tasks(self):
        if not self.datasets:
            raise ValueError("No datasets found")
        if "train" in self.datasets:
            return self.datasets.train.tasks
        return next(iter(self.datasets.values())).tasks

    @cached_property
    def dataset_tasks(self):
        if not self.datasets:
            raise ValueError("No datasets found")
        dataset = self.datasets.train if "train" in self.datasets else next(iter(self.datasets.values()))
        tasks = self.tasks
        dataset_tasks = dataset.dataset_tasks if isinstance(dataset, MultiTaskDataset) else dataset.tasks
        for dataset in self.datasets.values():
            if isinstance(dataset, MultiTaskDataset):
                for dataset_name, tasks_ in dataset.dataset_tasks.items():
                    for task_name, task_ in tasks_.items():
                        if task_name not in tasks:
                            raise ValueError(f"Task {task_name} of dataset {dataset_name} is not defined")
                        task = tasks[task_name]
                        if task != task_:
                            warn(
                                f"Task {task_name} of dataset {dataset_name} has different configurations "
                                "compared to training data, using training configuration.\n"
                                "This may lead to unexpected behavior.",
                            )
                        if dataset_name not in dataset_tasks:
                            dataset_tasks[dataset_name] = NestedDict()
                        if task_name not in dataset_tasks[dataset_name]:
                            dataset_tasks[dataset_name][task_name] = task
            else:
                for task_name, task_ in dataset.tasks.items():
                    if task_name not in tasks:
                        raise ValueError(f"Task {task_name} is not defined")
                    task = tasks[task_name]
                    if task != task_:
                        warn(
                            f"Task {task_name} has different configurations "
                            "compared to training data, using training configuration.\n"
                            "This may lead to unexpected behavior.",
                        )
                    if task_name not in dataset_tasks:
                        dataset_tasks[task_name] = task
        return dataset_tasks

    @cached_property
    def network(self):
        heads = {
            name: HeadConfig(num_labels=task.num_labels, problem_type=task.type, type=task.level)
            for name, task in self.tasks.items()
        }
        if "heads" not in self.config.network:
            self.config.network.heads = NestedDict(heads)
        else:
            self.config.network.heads.merge(heads, overwrite=False)
        return self.config.network

    def build_datasets(self):
        if "data" in self.config:
            self.datasets = self.all_datasets = self._build_dataset(self.config.data)
            return
        if "datas" in self.config:
            self.all_datasets = NestedDict(
                {name: self._build_dataset(config, name) for name, config in self.config.datas.items()}
            )
            datasets = {
                subkey: {key: subdict[subkey] for key, subdict in self.all_datasets.items() if subkey in subdict}
                for subkey in {k for v in self.all_datasets.values() for k in v}
            }
            self.datasets = NestedDict({split: MultiTaskDataset(datas) for split, datas in datasets.items()})
            return
        raise ValueError("No data configuration found")

    def _build_dataset(self, config: NestedDict, name: str | None = None) -> NestedDict:
        name = name or config.root
        print(f"Building dataset {name}")
        dataset = NestedDict()
        ignored_key = defaults.DATASET_SPLITS + ["root"]
        dataset_factory = partial(
            Dataset,
            tokenizer=self.tokenizer,
            **{k: v for k, v in config.items() if k not in ignored_key},
        )
        if os.path.isdir(config.root):
            if "train" in config:
                dataset.train = dataset_factory(os.path.join(config.root, config.train), split="train")
            if "validation" in config:
                dataset.validation = dataset_factory(os.path.join(config.root, config.validation), split="validation")
            if "test" in config:
                dataset.test = dataset_factory(os.path.join(config.root, config.test), split="test")
            if "evaluation" in config:
                dataset.evaluation = dataset_factory(os.path.join(config.root, config.evaluation), split="evaluation")
            if "inference" in config:
                dataset.inference = dataset_factory(os.path.join(config.root, config.inference), split="inference")
        else:
            splits = get_dataset_split_names(config.root)
            existing_splits = {k for k in defaults.DATASET_SPLITS if config.get(k) is not None}
            if not existing_splits:
                if "train" in splits:
                    config.train = "train"
                if "validation" in splits:
                    config.validation = "validation"
                if "test" in splits:
                    config.test = "test"
            if config.get("train") is not None:
                dataset.train = dataset_factory(config.root, split="train")
            if config.get("validation") is not None:
                dataset.validation = dataset_factory(config.root, split="validation")
            if config.get("test") is not None:
                dataset.test = dataset_factory(config.root, split="test")
            if config.get("evaluation") is not None:
                dataset.evaluation = dataset_factory(config.root, split=config.get("evaluation"))
            if config.get("inference") is not None:
                dataset.inference = dataset_factory(config.root, split=config.get("inference"))
        if not dataset:
            raise ValueError("No datasets built. This is likely due to missing data paths in Config.")
        return dataset

    def build_dataloaders(self):
        datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
        default_kwargs = self.config.get("dataloader", NestedDict())
        dataloader_kwargs = NestedDict({k: default_kwargs.pop(k) for k in self.datasets if k in default_kwargs})
        for k, d in datasets.items():
            dataloader_kwargs.setdefault(k, NestedDict())
            dataloader_kwargs[k].merge(default_kwargs, overwrite=False)
            batch_size = dataloader_kwargs[k].pop("batch_size")
            shuffle = dataloader_kwargs[k].pop("shuffle", getattr(d, "train", True))
            drop_last = dataloader_kwargs[k].pop("drop_last", not getattr(d, "train", True))
            if isinstance(d, MultiTaskDataset):
                batch_sampler = (
                    DistributedMultiTaskSampler(d, batch_size, shuffle=shuffle, drop_last=drop_last)
                    if self.distributed
                    else MultiTaskSampler(d, batch_size, shuffle=shuffle, drop_last=drop_last)
                )
            else:
                sampler = (
                    data.distributed.DistributedSampler(d, shuffle=shuffle)
                    if self.distributed
                    else data.RandomSampler(d) if shuffle else data.SequentialSampler(d)
                )
                batch_sampler = data.BatchSampler(sampler, batch_size, drop_last=drop_last)
            self.dataloaders[k] = data.DataLoader(
                d, batch_sampler=batch_sampler, collate_fn=self.collate_fn, **dataloader_kwargs[k]
            )

    def build_metrics(self) -> MultiTaskMetrics:
        return MultiTaskMetrics(
            {
                name: MetricRegistry.build(type=task.type, num_labels=task.num_labels)
                for name, task in self.dataset_tasks.all_items()
            }
        )

    def collate_fn(self, batch):
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

    def get_dataset_lengths(self) -> str:
        repr = "dataset lengths:\n"
        longest_name = max(len(name) for name in self.all_datasets.keys())
        for name, dataset in self.all_datasets.items():
            if isinstance(dataset, NestedDict):
                repr += f"{name}:"
                if len(name) < longest_name:
                    repr += " " * (longest_name - len(name))
                repr += "\t\t"
                for split, d in dataset.items():
                    repr += f"  {split}: {len(d)}\t"
            else:
                repr += f"{name}: {len(dataset)}\t"
            repr += "\n"
        return repr
