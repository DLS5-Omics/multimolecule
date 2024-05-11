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

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from random import choices

import torch
from chanfig import NestedDict
from torch import distributed as dist
from torch.utils import data

from .dataset import Dataset


class MultiTaskDataset(data.ConcatDataset):

    datasets: Mapping
    dataset_keys: Sequence[str]
    dataset_values: Sequence[Dataset]

    def __init__(self, datasets: Mapping) -> None:
        for key, dataset in datasets.items():
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Dataset {key} should be an instance of Dataset")
        self.datasets = datasets
        if not len(self.datasets) > 0:
            raise ValueError("MultiTaskDataset should contain at least one dataset")
        self.dataset_keys, self.dataset_values = zip(*self.datasets.items())
        self.cumulative_sizes = self.cumsum(self.dataset_values)

    def __getitems__(self, key: Sequence[int]) -> Mapping:
        dataset_idx = bisect_right(self.cumulative_sizes, key[0])
        if dataset_idx == 0:
            sample_idx = key
        else:
            sample_idx = [i - self.cumulative_sizes[dataset_idx - 1] for i in key]
        batch = self.dataset_values[dataset_idx][sample_idx]
        batch["dataset"] = self.dataset_keys[dataset_idx]
        return batch

    @property
    def tasks(self) -> NestedDict:
        tasks = NestedDict()
        for dataset in self.dataset_values:
            for n, t in dataset.tasks.items():
                if n not in tasks:
                    tasks[n] = t
                elif tasks[n] != t:
                    raise ValueError(f"Task {n} has different configurations across datasets")
        return tasks

    @property
    def dataset_tasks(self) -> NestedDict:
        return NestedDict({k: v.tasks for k, v in self.datasets.items()})

    def __repr__(self) -> str:
        return f"MultiTaskDataset({', '.join([str(d) for d in self.datasets])})"


class MultiTaskSampler(data.BatchSampler):
    r"""
    Ensure all items in a batch comes from the same dataset.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    datasets: Sequence[Dataset]

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dataset: MultiTaskDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler_cls: type[data.Sampler] | None = None,
        weights: list[int] | None = None,
    ) -> None:
        self.datasets = dataset.dataset_values
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if sampler_cls is None:
            sampler_cls = data.RandomSampler if shuffle else data.SequentialSampler
        self.samplers = [sampler_cls(d) for d in self.datasets]  # type: ignore
        self.dataset_sizes = [len(d) for d in self.datasets]  # type: ignore
        self.cumulative_sizes = dataset.cumulative_sizes
        self.num_datasets = len(self.datasets)
        self.weights = weights if weights is not None else self.dataset_sizes

    def __iter__(self):
        sampler_iters = [(i, iter(s)) for i, s in enumerate(self.samplers)]
        sampler_weights = deepcopy(self.weights)
        sampler_idx = 0
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            while sampler_iters:
                if self.shuffle:
                    sampler_idx = choices(range(len(sampler_iters)), weights=sampler_weights)[0]
                sampler_id, sampler_iter = sampler_iters[sampler_idx]
                cumulative_size = self.cumulative_sizes[sampler_id - 1] if sampler_id > 0 else 0
                try:
                    batch = [next(sampler_iter) + cumulative_size for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    sampler_iters.pop(sampler_idx)
                    sampler_weights.pop(sampler_idx)
        else:
            while sampler_iters:
                if self.shuffle:
                    sampler_idx = choices(range(len(sampler_iters)), weights=sampler_weights)[0]
                sampler_id, sampler_iter = sampler_iters[sampler_idx]
                cumulative_size = self.cumulative_sizes[sampler_id - 1] if sampler_id > 0 else 0
                batch = [0] * self.batch_size
                idx_in_batch = 0
                try:
                    for _ in range(self.batch_size):
                        batch[idx_in_batch] = next(sampler_iter) + cumulative_size
                        idx_in_batch += 1
                    yield batch
                    idx_in_batch = 0  # noqa: SIM113
                    batch = [0] * self.batch_size
                except StopIteration:
                    sampler_iters.pop(sampler_idx)
                    sampler_weights.pop(sampler_idx)
                    if idx_in_batch > 0:
                        yield batch[:idx_in_batch]

    def __len__(self):
        batch_size = self.batch_size
        if self.drop_last:
            return sum(len(d) // batch_size for d in self.datasets)
        return sum((len(d) + batch_size - 1) // batch_size for d in self.datasets)


class DistributedMultiTaskSampler(MultiTaskSampler):  # pylint: disable=too-few-public-methods
    r"""
    Distributed version of MultiTaskSampler, which ensures that all GPUs sample data from the
    same sub-dataset in each step without requiring additional communication.
    The dataset selection is based on a random seed mechanism that is synchronized across epochs.

    See Also:
        [MultiTaskSampler][MultiTaskSampler]
    """

    def __init__(
        self,
        dataset: MultiTaskDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler_cls: type[data.Sampler] = data.RandomSampler,
        weights: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, batch_size, shuffle, drop_last, sampler_cls, weights)
        self.samplers = [data.DistributedSampler(d, shuffle=shuffle, drop_last=drop_last) for d in self.datasets]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """
        Sets the epoch for deterministic shuffling.
        """
        self.epoch = epoch
        for sampler in self.samplers:
            sampler.set_epoch(epoch)

    def _get_sampler_idx(self, high: int) -> int:
        """
        Determines which sampler (i.e., sub-dataset) to use based on the seed and epoch.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampler_idx = torch.randint(low=0, high=high, size=(1,), generator=g).item()
        return sampler_idx

    def __iter__(self) -> Iterator:
        sampler_iters = [(i, iter(s)) for i, s in enumerate(self.samplers)]
        sampler_weights = deepcopy(self.weights)

        if self.drop_last:
            while sampler_iters:
                # Sample the same sub-dataset across all GPUs using the seeded index
                sampler_idx = self._get_sampler_idx(len(sampler_iters))
                sampler_id, sampler_iter = sampler_iters[sampler_idx]
                cumulative_size = self.cumulative_sizes[sampler_id - 1] if sampler_id > 0 else 0
                try:
                    batch = [next(sampler_iter) + cumulative_size for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    sampler_iters.pop(sampler_idx)
                    sampler_weights.pop(sampler_idx)
        else:
            while sampler_iters:
                # Sample the same sub-dataset across all GPUs using the seeded index
                sampler_idx = self._get_sampler_idx(len(sampler_iters))
                sampler_id, sampler_iter = sampler_iters[sampler_idx]
                cumulative_size = self.cumulative_sizes[sampler_id - 1] if sampler_id > 0 else 0
                batch = [0] * self.batch_size
                idx_in_batch = 0
                try:
                    for _ in range(self.batch_size):
                        batch[idx_in_batch] = next(sampler_iter) + cumulative_size
                        idx_in_batch += 1
                    yield batch
                    idx_in_batch = 0  # noqa: SIM113
                    batch = [0] * self.batch_size
                except StopIteration:
                    sampler_iters.pop(sampler_idx)
                    sampler_weights.pop(sampler_idx)
                    if idx_in_batch > 0:
                        yield batch[:idx_in_batch]

    def __len__(self) -> int:
        batch_size = self.batch_size * self.world_size
        if self.drop_last:
            return sum(len(d) // batch_size for d in self.datasets)
        return sum((len(d) + batch_size - 1) // batch_size for d in self.datasets)

    @property
    def world_size(self) -> int:
        r"""Return the number of processes in the current process group."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1
