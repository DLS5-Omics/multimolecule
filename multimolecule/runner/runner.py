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
from functools import cached_property, partial
from typing import Any, Tuple

import danling as dl
import torch
from art import text2art
from chanfig import FlatDict, NestedDict
from danling import MetricMeters
from datasets import disable_progress_bars, get_dataset_split_names
from huggingface_hub.utils import validate_repo_id
from lazy_imports import try_import
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer

from multimolecule import defaults
from multimolecule.data import Dataset, build_dataset
from multimolecule.module import HeadConfig, ModelRegistry, MultiMoleculeModel

from .config import MultiMoleculeConfig
from .metrics import MetricRegistry

with try_import() as ema:
    from ema_pytorch import EMA


disable_progress_bars()


class MultiMoleculeRunner(dl.Runner):

    config: MultiMoleculeConfig
    model: MultiMoleculeModel
    raw_datasets: NestedDict

    def __init__(self, config: MultiMoleculeConfig):
        if config.art:
            print(text2art("MultiMolecule", "rand-large"))
        if not config.pretrained:
            raise ValueError("A pretrained model must be specified via config.pretrained")
        # We do not want to check if it actually exists, because it might be a local directory
        validate_repo_id(config.pretrained)
        super().__init__(config)
        # must build tokenizer before datasets
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained)
        self.datasets = self.build_datasets()
        self.dataloaders = self.build_dataloaders()
        self.model = ModelRegistry.build(**self.network)
        self.model = self.model.to(self.device)
        ema_enabled = self.config.ema.pop("enabled", False)
        if ema_enabled:
            ema.check()
            self.ema = EMA(self.model, **self.config.ema)
        self.config.ema.enabled = ema_enabled
        if self.config.training:
            optim_name = self.config.optim.pop("name", "AdamW")
            pretrained_ratio = self.config.optim.pop("pretrained_ratio", 1e-2)
            self.optimizer = self.get_optimizer(optim_name)(
                self.model.trainable_parameters(pretrained_ratio=pretrained_ratio, **self.config.optim),
                **self.config.optim,
            )
            self.config.optim.name = optim_name
            self.config.optim.pretrained_ratio = pretrained_ratio
            if self.config.sched:
                self.scheduler = dl.optim.LRScheduler(self.optimizer, total_steps=self.total_steps, **self.config.sched)
        self.train_metrics = self.build_train_metrics()
        self.evaluate_metrics = self.build_evaluate_metrics()

    def __post_init__(self):
        if self.config.platform != "deepspeed" and "checkpoint" in self.config:
            self.load_checkpoint(self.config.checkpoint)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, find_unused_parameters=True, bucket_cap_mb=32, gradient_as_bucket_view=True
            )
        super().__post_init__()
        if self.config.platform == "deepspeed" and "checkpoint" in self.config:
            self.load_checkpoint(self.config.checkpoint)
        self.yaml(os.path.join(self.dir, "trainer.yaml"))
        print(self)

    def train_step(self, data) -> Tuple[Any, torch.Tensor]:
        with self.autocast(), self.accumulate():
            pred = self.model(**data)
            self.advance(pred["loss"])
            self.metric_fn(pred, data)
        return pred, pred["loss"]

    def evaluate_step(self, data) -> Tuple[Any, torch.Tensor]:
        model = self.ema or self.model
        output = model(**data)
        self.metric_fn(output, data)
        return output, output["loss"]

    @torch.inference_mode()
    def infer(self, split: str = "inf") -> NestedDict | FlatDict | list:
        r"""
        Perform inference on `split`.

        Args:
            split (str): split to run inference

        Return:
            Inference outputs.

            - If the model has single output:

                - If labels are available, a [`FlatDict`][chanfig.FlatDict] with keys `predict` and `label` is
                    returned.
                - If labels are not available, a list of predictions is returned.

            - If the model has multiple outputs:
                - If labels are available, a [`NestedDict`][chanfig.NestedDict] with keys as task names and values
                    as dictionaries with keys `predict` and `label` is returned.
                - If labels are not available, a [`FlatDict`][chanfig.FlatDict] with keys as task names and values
                    as lists of predictions is returned.
        """

        self.mode = "inf"  # type: ignore
        loader = self.dataloaders[split]
        preds = FlatDict()
        labels = FlatDict()
        model = self.ema or self.model
        for _, data in tqdm(enumerate(loader), total=len(loader)):  # noqa: F402
            pred = model(**data)
            if isinstance(pred, tuple):
                pred, loss = pred
            for task, p in pred.items():
                preds[task].extend(p["logits"].squeeze(-1).tolist())
                if task in data:
                    labels[task].extend(data[task].squeeze(-1).tolist())

        if self.distributed:
            torch.cuda.synchronize()
            for task in preds.keys():
                preds[task] = self.gather_for_metrics(preds[task])
            for task in labels.keys():
                labels[task] = self.gather_for_metrics(labels[task])
        if labels:
            if len(preds) == 1:
                return FlatDict(predict=next(iter(preds.values())), label=next(iter(labels.values())))
            return NestedDict({task: {"predict": preds[task], "label": labels[task]} for task in preds})
        if len(preds) == 1:
            return next(iter(preds.values()))
        return preds

    def metric_fn(self, pred, data):
        self.metrics.update(pred["logits"], data["labels"])
        self.meters.update({"loss": pred["loss"]})

    @cached_property
    def task(self):
        if not self.datasets:
            raise ValueError("No datasets found")
        if "train" in self.datasets:
            return self.datasets.train.task
        return next(iter(self.datasets.values())).task

    @cached_property
    def network(self):
        head = HeadConfig(num_labels=self.task.num_labels, problem_type=self.task.type, type=self.task.level)
        if "head" not in self.config.network:
            self.config.network.head = NestedDict(head)
        else:
            self.config.network.head.merge(head, overwrite=False)
        return self.config.network

    def build_datasets(self) -> NestedDict[str, Dataset]:
        return self._build_dataset(self.config.data)

    def _build_dataset(self, config: NestedDict, name: str | None = None) -> NestedDict:
        root = config.pop("root", None)
        ratio = config.pop("ratio", None)
        name = name or root

        is_local_dataset = os.path.isdir(root)
        try:
            is_hf_dataset = validate_repo_id(root) is None
        except Exception:
            is_hf_dataset = False

        dataset = NestedDict()
        if is_local_dataset:
            train_splits = [key for key in config.keys() if key.startswith(defaults.TRAIN_SPLITS)]
            other_splits = [
                key for key in config.keys() if key.startswith(defaults.DATASET_SPLITS) and key not in train_splits
            ]
            print(f"Building local dataset {name}")
        elif is_hf_dataset:
            splits = [k for k in defaults.DATASET_SPLITS if config.get(k) is not None] or get_dataset_split_names(root)
            train_splits = [key for key in splits if key.startswith(defaults.TRAIN_SPLITS)]
            other_splits = [key for key in splits if key not in train_splits]
            print(f"Building HuggingFace dataset {name}")
        else:
            raise ValueError(
                f"Dataset root '{root}' is invalid. It must be either:\n"
                f"  - A path to an existing local directory\n"
                f"  - A valid Hugging Face dataset repository ID"
            )

        ignored_keys = train_splits + other_splits
        dataset_factory = partial(
            build_dataset,
            tokenizer=self.tokenizer,
            **{k: v for k, v in config.items() if k not in ignored_keys},
        )

        if is_local_dataset:
            for split in train_splits:
                dataset[split] = dataset_factory(
                    os.path.join(root, config[split]), split=split, train=True, ratio=ratio
                )
            for split in other_splits:
                dataset[split] = dataset_factory(os.path.join(root, config[split]), split=split, train=False)
        elif is_hf_dataset:
            for split in train_splits:
                dataset[split] = dataset_factory(root, split=split, train=True, ratio=ratio)
            for split in other_splits:
                dataset[split] = dataset_factory(root, split=split, train=False)

        if not dataset:
            raise ValueError(f"No datasets built. This is likely due to missing data paths in {config}.")

        config.root = root
        config.ratio = ratio
        return dataset

    def build_dataloaders(self) -> NestedDict[str, data.DataLoader]:
        dataloaders = NestedDict()
        datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
        default_kwargs = self.config.get("dataloader", NestedDict())
        dataloader_kwargs = NestedDict({k: default_kwargs.pop(k) for k in self.datasets if k in default_kwargs})
        for k, d in datasets.items():
            dataloader_kwargs.setdefault(k, NestedDict())
            dataloader_kwargs[k].merge(default_kwargs, overwrite=False)
            batch_size = dataloader_kwargs[k].pop("batch_size")
            shuffle = dataloader_kwargs[k].pop("shuffle", getattr(d, "train", True))
            drop_last = dataloader_kwargs[k].pop("drop_last", not getattr(d, "train", True))
            sampler = (
                data.distributed.DistributedSampler(d, shuffle=shuffle)
                if self.distributed
                else data.RandomSampler(d) if shuffle else data.SequentialSampler(d)
            )
            batch_sampler = data.BatchSampler(sampler, batch_size, drop_last=drop_last)
            dataloaders[k] = data.DataLoader(
                d, batch_sampler=batch_sampler, collate_fn=self.collate_fn, **dataloader_kwargs[k]
            )
        return dataloaders

    def build_train_metrics(self) -> MetricMeters:
        return MetricRegistry.build(type=self.task.type, num_labels=self.task.num_labels)

    def build_evaluate_metrics(self) -> MetricMeters:
        return MetricRegistry.build(type=self.task.type, num_labels=self.task.num_labels)

    def collate_fn(self, batch):
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
