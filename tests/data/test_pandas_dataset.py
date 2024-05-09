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

import os
from functools import partial

import danling as dl
import pytest
import torch

from multimolecule import PandasDataset


@pytest.mark.lfs
class TestRNADataset:

    pretrained = "multimolecule/rna"
    root = os.path.join("tests", "data", "datasets", "rna")

    @pytest.mark.parametrize("preprocess", [True, False])
    def test_5utr(self, preprocess: bool):
        file = os.path.join(self.root, "5utr.csv")
        dataset = PandasDataset(
            file, split="train", pretrained=self.pretrained, preprocess=preprocess, auto_rename_cols=True
        )
        elem = dataset[0]
        assert isinstance(elem["input_ids"], dl.PNTensor)
        assert isinstance(elem["labels"], torch.FloatTensor)
        batch = dataset[list(range(3))]
        assert isinstance(batch["input_ids"], dl.NestedTensor)
        assert isinstance(batch["labels"], torch.FloatTensor)

    @pytest.mark.parametrize("preprocess", [True, False])
    def test_ncrna(self, preprocess: bool):
        file = os.path.join(self.root, "ncrna.csv")
        dataset = PandasDataset(
            file, split="train", pretrained=self.pretrained, preprocess=preprocess, auto_rename_cols=True
        )
        elem = dataset[0]
        assert isinstance(elem["input_ids"], dl.PNTensor)
        assert isinstance(elem["labels"], torch.LongTensor)
        batch = dataset[list(range(3))]
        assert isinstance(batch["input_ids"], dl.NestedTensor)
        assert isinstance(batch["labels"], torch.LongTensor)

    @pytest.mark.parametrize("preprocess", [True, False])
    def test_rnaswitches(self, preprocess: bool):
        file = os.path.join(self.root, "rnaswitches.csv")
        label_cols = ["ON", "OFF", "ON_OFF"]
        dataset = PandasDataset(
            file, split="train", pretrained=self.pretrained, preprocess=preprocess, label_cols=label_cols
        )
        elem = dataset[0]
        assert isinstance(elem["sequence"], dl.PNTensor)
        assert isinstance(elem["ON"], torch.FloatTensor)
        assert isinstance(elem["OFF"], torch.FloatTensor)
        batch = dataset[list(range(3))]
        assert isinstance(batch["sequence"], dl.NestedTensor)
        assert isinstance(batch["ON_OFF"], torch.FloatTensor)

    @pytest.mark.parametrize("preprocess", [True, False])
    def test_modifications(self, preprocess: bool):
        file = os.path.join(self.root, "modifications.json")
        dataset = PandasDataset(file, split="train", pretrained=self.pretrained, preprocess=preprocess)
        elem = dataset[0]
        assert isinstance(elem["sequence"], dl.PNTensor)
        assert isinstance(elem["label"], torch.LongTensor)
        batch = dataset[list(range(3))]
        assert isinstance(batch["sequence"], dl.NestedTensor)
        assert isinstance(batch["label"], torch.LongTensor)

    @pytest.mark.parametrize("preprocess", [True, False])
    def test_degradation(self, preprocess: bool):
        file = os.path.join(self.root, "degradation.json")
        feature_cols = ["sequence"]  # , "structure", "predicted_loop_type"]
        label_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]
        dataset = PandasDataset(
            file,
            split="train",
            pretrained=self.pretrained,
            preprocess=preprocess,
            feature_cols=feature_cols,
            label_cols=label_cols,
        )
        elem = dataset[0]
        assert isinstance(elem["sequence"], dl.PNTensor)
        assert isinstance(elem["deg_pH10"], torch.FloatTensor)
        assert isinstance(elem["deg_50C"], torch.FloatTensor)
        batch = dataset[list(range(3))]
        assert isinstance(batch["sequence"], dl.NestedTensor)
        assert isinstance(batch["reactivity"], torch.FloatTensor)


@pytest.mark.lfs
class TestSyntheticDataset:

    pretrained = "multimolecule/rna"
    root = os.path.join("tests", "data", "datasets", "synthetic")

    def test_null(self):
        file = os.path.join(self.root, "null.csv")
        dataset_factory = partial(PandasDataset, file, split="train", pretrained=self.pretrained)
        dataset = dataset_factory(nan_process="ignore")
        assert len(dataset) == 67
        with pytest.raises(RuntimeError):
            dataset[0]
        with pytest.raises(ValueError):
            dataset = dataset_factory(nan_process="raise")
        dataset = dataset_factory(nan_process="fill", fill_value=0)
        assert dataset[0]["label"] == 0
        dataset = dataset_factory(nan_process="fill", fill_value=1)
        assert dataset[0]["label"] == 1
        dataset = dataset_factory(nan_process="drop")
        assert len(dataset) == 61
