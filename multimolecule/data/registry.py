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

from chanfig import Registry as Registry_
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class Registry(Registry_):  # pylint: disable=too-few-public-methods
    def build(
        self, *args, ratio: float | int | None = None, type: str | None = None, **kwargs
    ) -> TorchDataset | HFDataset:
        dataset = super().build(type, *args, **kwargs)
        if ratio is not None:
            from .dataset import SampleDataset

            dataset = SampleDataset(dataset, ratio=ratio)
        return dataset


DATASETS = Registry()

__all__ = ["DATASETS"]
