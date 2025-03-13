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

from typing import Callable

import datasets
from danling.utils import get_world_size
from torch.utils import data

from multimolecule import defaults

from .utils import no_collate


class DataLoader(data.DataLoader):

    def __init__(
        self,
        dataset: data.Dataset | datasets.Dataset,
        batch_size: int,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        distributed: bool | None = None,
        collate_fn: Callable = no_collate,
        **kwargs,
    ):
        is_train = None
        if hasattr(dataset, "train"):
            is_train = dataset.train
        elif hasattr(dataset, "split"):
            is_train = dataset.split in defaults.TRAIN_SPLITS
        if shuffle is None:
            if is_train is None:
                raise ValueError("`shuffle` must be specified if dataset is not a MultiMolecule Dataset")
            shuffle = is_train
        if drop_last is None:
            drop_last = False
        if distributed is None:
            distributed = get_world_size() > 1
        sampler = (
            data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if distributed
            else data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)
        )
        batch_sampler = data.BatchSampler(sampler, batch_size, drop_last=drop_last)
        super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, batch_size={self._batch_size}, shuffle={self._shuffle}, drop_last={self._drop_last}, num_workers={self.num_workers}, pin_memory={self.pin_memory})"  # noqa: E501
