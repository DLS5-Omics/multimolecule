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

from collections.abc import Mapping
from typing import List
from warnings import warn

import danling as dl
import datasets
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from .dataset import Dataset


class PandasDataset(Dataset):
    r"""
    A dataset from a [`pandas.DataFrame`][].

    Args:
        dataframe (DataFrame | str): A [`pandas.DataFrame`][] or a path to data file or a [`dict`][] of data.
        split (str): The split of the dataset.

    See Also:
        [`Dataset.post`][multimolecule.Dataset.post]

    """

    def __init__(
        self,
        dataframe: DataFrame | dict | str,
        split: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        pretrained: str | None = None,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        preprocess: bool = True,
        column_names_map: Mapping[str, str] | None = None,
        auto_rename_cols: bool = False,
        nan_process: str | None = "drop",
        fill_value: str | int | float = 0,
        truncation: bool | None = None,
        max_length: int | None = None,
    ):
        if isinstance(dataframe, str):
            dataframe = dl.load_pandas(dataframe)
        if isinstance(dataframe, dict):
            dataframe = DataFrame.from_dict(dataframe)
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
        if feature_cols is not None and label_cols is not None:
            dataframe = dataframe[feature_cols + label_cols]  # type: ignore[operator]
        dataframe = self.process_nan(dataframe, nan_process=nan_process, fill_value=fill_value)
        table = datasets.table.InMemoryTable.from_pandas(dataframe, preserve_index=False)
        super().__init__(
            table,
            split=split,
            tokenizer=tokenizer,
            pretrained=pretrained,
            feature_cols=feature_cols,
            label_cols=label_cols,
            preprocess=preprocess,
            column_names_map=column_names_map,
            auto_rename_cols=auto_rename_cols,
            truncation=truncation,
            max_length=max_length,
        )

    def process_nan(
        self, dataframe: DataFrame, nan_process: str | None, fill_value: str | int | float = 0
    ) -> DataFrame:
        dataframe = dataframe.replace([float("inf"), -float("inf")], float("nan"))
        if nan_process == "ignore":
            return dataframe
        if dataframe.isnull().values.any():
            if nan_process is None or nan_process == "error":
                raise ValueError("NaN / inf values have been found in the dataset.")
            warn(
                "NaN / inf values have been found in the dataset.\n"
                "While we can handle them, the data type of the corresponding column may be set to float, "
                "which can and very likely will disrupt the auto task recognition.\n"
                "It is recommended to address these values before loading the dataset."
            )
            if nan_process == "drop":
                return dataframe.dropna()
            if nan_process == "fill":
                return dataframe.fillna(fill_value)
            raise ValueError(f"Invalid nan_process: {nan_process}")
        return dataframe
