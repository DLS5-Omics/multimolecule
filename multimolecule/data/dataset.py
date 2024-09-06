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

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, List
from warnings import warn

import danling as dl
import datasets
import pyarrow as pa
import torch
from chanfig import NestedDict
from danling import NestedTensor
from datasets.table import Table
from pandas import DataFrame
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from multimolecule import defaults
from multimolecule.tasks import Task, TaskLevel, TaskType

from .functional import dot_bracket_to_contact_map
from .utils import infer_discrete_map, infer_task, map_value, truncate_value

# from multimolecule.tokenisers.dot_bracket.utils import STANDARD_ALPHABET as DOT_BRACKET_ALPHABET


class Dataset(datasets.Dataset):
    r"""
    The base class for all datasets.

    Dataset is a subclass of [`datasets.Dataset`][] that provides additional functionality for handling structured data.
    It has three main features:

    - column identification: identify the special columns (sequence and structure columns) in the dataset.
    - tokenization: tokenize the sequence columns in the dataset using a pretrained tokenizer.
    - task inference: infer the task type and level of each label column in the dataset.

    Attributes:
        tasks: A nested dictionary of the inferred tasks for each label column in the dataset.
        tokenizer: The pretrained tokenizer to use for tokenization.
        truncation: Whether to truncate sequences that exceed the maximum length of the tokenizer.
        max_seq_length: The maximum length of the input sequences.
        data_cols: The names of all columns in the dataset.
        feature_cols: The names of the feature columns in the dataset.
        label_cols: The names of the label columns in the dataset.
        sequence_cols: The names of the sequence columns in the dataset.
        column_names_map: A mapping of column names to new column names.
        preprocess: Whether to preprocess the dataset.

    Args:
        data: The dataset. This can be a path to a file, a tag on the Hugging Face Hub, a pyarrow.Table,
            a [dict][], a [list][], or a [pandas.DataFrame][].
        split: The split of the dataset.
        tokenizer: A pretrained tokenizer to use for tokenization.
            Either `tokenizer` or `pretrained` must be specified.
        pretrained: The name of a pretrained tokenizer to use for tokenization.
            Either `tokenizer` or `pretrained` must be specified.
        feature_cols: The names of the feature columns in the dataset.
            Will be inferred automatically if not specified.
        label_cols: The names of the label columns in the dataset.
            Will be inferred automatically if not specified.
        id_cols: The names of the ID columns in the dataset.
            Will be inferred automatically if not specified.
        preprocess: Whether to preprocess the dataset.
            Preprocessing involves pre-tokenizing the sequences using the tokenizer.
            Defaults to `True`.
        auto_rename_cols: Whether to automatically rename columns to standard names.
            Only works when there is exactly one feature column / one label column.
            You can control the naming through `multimolecule.defaults.SEQUENCE_COL_NAME` and
            `multimolecule.defaults.LABEL_COL_NAME`.
            For more refined control, use `column_names_map`.
        column_names_map: A mapping of column names to new column names.
            This is useful for renaming columns to inputs that are expected by a model.
            Defaults to `None`.
        truncation: Whether to truncate sequences that exceed the maximum length of the tokenizer.
            Defaults to `False`.
        max_seq_length: The maximum length of the input sequences.
            Defaults to the `model_max_length` of the tokenizer.
        tasks: A mapping of column names to tasks.
            Will be inferred automatically if not specified.
        discrete_map: A mapping of column names to discrete mappings.
            This is useful for mapping the raw value to nominal value in classification tasks.
            Will be inferred automatically if not specified.
        nan_process: How to handle NaN and inf values in the dataset.
            Can be "ignore", "error", "drop", or "fill". Defaults to "ignore".
        fill_value: The value to fill NaN and inf values with.
            Defaults to 0.
        info: The dataset info.
        indices_table: The indices table.
        fingerprint: The fingerprint of the dataset.
    """

    tokenizer: PreTrainedTokenizerBase
    truncation: bool = False
    max_seq_length: int
    seq_length_offset: int = 0

    _id_cols: List
    _feature_cols: List
    _label_cols: List

    _sequence_cols: List
    _secondary_structure_cols: List

    _tasks: NestedDict[str, Task]
    _discrete_map: Mapping

    preprocess: bool = True
    auto_rename_cols: bool = False
    column_names_map: Mapping[str, str] | None = None
    ignored_cols: List[str] = []

    def __init__(
        self,
        data: Table | DataFrame | dict | list | str,
        split: datasets.NamedSplit,
        tokenizer: PreTrainedTokenizerBase | None = None,
        pretrained: str | None = None,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        id_cols: List | None = None,
        preprocess: bool | None = None,
        auto_rename_cols: bool | None = None,
        column_names_map: Mapping[str, str] | None = None,
        truncation: bool | None = None,
        max_seq_length: int | None = None,
        tasks: Mapping[str, Task] | None = None,
        discrete_map: Mapping[str, int] | None = None,
        nan_process: str = "ignore",
        fill_value: str | int | float = 0,
        info: datasets.DatasetInfo | None = None,
        indices_table: Table | None = None,
        fingerprint: str | None = None,
        ignored_cols: List[str] | None = None,
    ):
        if tasks is not None:
            self._tasks = NestedDict(tasks)
        if discrete_map is not None:
            self._discrete_map = discrete_map
        arrow_table = self.build_table(
            data, split, feature_cols, label_cols, nan_process=nan_process, fill_value=fill_value
        )
        super().__init__(
            arrow_table=arrow_table, split=split, info=info, indices_table=indices_table, fingerprint=fingerprint
        )
        self.identify_special_cols(feature_cols=feature_cols, label_cols=label_cols, id_cols=id_cols)
        self.post(
            tokenizer=tokenizer,
            pretrained=pretrained,
            preprocess=preprocess,
            truncation=truncation,
            max_seq_length=max_seq_length,
            auto_rename_cols=auto_rename_cols,
            column_names_map=column_names_map,
        )
        self.ignored_cols = ignored_cols or self.id_cols

    def build_table(
        self,
        data: Table | DataFrame | dict | str,
        split: datasets.NamedSplit,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        nan_process: str | None = "ignore",
        fill_value: str | int | float = 0,
    ) -> datasets.table.Table:
        if isinstance(data, str):
            try:
                data = datasets.load_dataset(data, split=split).data
            except FileNotFoundError:
                data = dl.load_pandas(data)
                if isinstance(data, DataFrame):
                    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
                    data = pa.Table.from_pandas(data)
        elif isinstance(data, dict):
            data = pa.Table.from_pydict(data)
        elif isinstance(data, list):
            data = pa.Table.from_pylist(data)
        elif isinstance(data, DataFrame):
            data = pa.Table.from_pandas(data)
        if feature_cols is not None and label_cols is not None:
            data = data.select(feature_cols + label_cols)
        data = self.process_nan(data, nan_process=nan_process, fill_value=fill_value)
        return data

    def post(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        pretrained: str | None = None,
        max_seq_length: int | None = None,
        truncation: bool | None = None,
        preprocess: bool | None = None,
        auto_rename_cols: bool | None = None,
        column_names_map: Mapping[str, str] | None = None,
    ) -> None:
        r"""
        Perform pre-processing steps after initialization.

        It first identifies the special columns (sequence and structure columns) in the dataset.
        Then it sets the feature and label columns based on the input arguments.
        If `auto_rename_cols` is `True`, it will automatically rename the columns to model inputs.
        Finally, it sets the [`transform`][datasets.Dataset.set_transform] function based on the `preprocess` flag.
        """
        if tokenizer is None:
            if pretrained is None:
                raise ValueError("tokenizer and pretrained can not be both None.")
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        if max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
        else:
            tokenizer.model_max_length = max_seq_length
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if truncation is not None:
            self.truncation = truncation
        if self.tokenizer.cls_token is not None:
            self.seq_length_offset += 1
        if self.tokenizer.sep_token is not None and self.tokenizer.sep_token != self.tokenizer.eos_token:
            self.seq_length_offset += 1
        if self.tokenizer.eos_token is not None:
            self.seq_length_offset += 1
        if preprocess is not None:
            self.preprocess = preprocess
        if auto_rename_cols is not None:
            self.auto_rename_cols = auto_rename_cols
        if self.auto_rename_cols:
            if column_names_map is not None:
                raise ValueError("auto_rename_cols and column_names_map are mutually exclusive.")
            column_names_map = {}
            if len(self.feature_cols) == 1:
                column_names_map[self.feature_cols[0]] = defaults.SEQUENCE_COL_NAME
            if len(self.label_cols) == 1:
                column_names_map[self.label_cols[0]] = defaults.LABEL_COL_NAME
        self.column_names_map = column_names_map
        if self.column_names_map:
            self.rename_columns(self.column_names_map)

        if self.preprocess:
            self.update(self.map(self.tokenization))
            if self.secondary_structure_cols:
                self.update(self.map(self.convert_secondary_structure))
            if self.discrete_map:
                self.update(self.map(self.map_discrete))
            fn_kwargs = {
                "columns": [name for name, task in self.tasks.items() if task.level in ["nucleotide", "contact"]],
                "max_seq_length": self.max_seq_length - self.seq_length_offset,
            }
            if self.truncation and 0 < self.max_seq_length < 2**32:
                self.update(self.map(self.truncate, fn_kwargs=fn_kwargs))
        self.set_transform(self.transform)

    def transform(self, batch: Mapping) -> Mapping:
        r"""
        Default [`transform`][datasets.Dataset.set_transform].

        See Also:
            [`collate`][multimolecule.Dataset.collate]
        """
        return {k: self.collate(k, v) for k, v in batch.items() if k not in self.ignored_cols}

    def collate(self, col: str, data: Any) -> Tensor | NestedTensor | None:
        r"""
        Collate the data for a column.

        If the column is a sequence column, it will tokenize the data if `tokenize` is `True`.
        Otherwise, it will return a tensor or nested tensor.
        """
        if col in self.sequence_cols:
            if isinstance(data[0], str):
                data = self.tokenize(data)
            return NestedTensor(data)
        if not self.preprocess:
            if col in self.discrete_map:
                data = map_value(data, self.discrete_map[col])
            if col in self.tasks:
                data = truncate_value(data, self.max_seq_length - self.seq_length_offset, self.tasks[col].level)
        if isinstance(data[0], str):
            return data
        try:
            return torch.tensor(data)
        except ValueError:
            return NestedTensor(data)

    def infer_tasks(self, tasks: Mapping | None = None, sequence_col: str | None = None) -> NestedDict:
        self._tasks = tasks or NestedDict()
        for col in self.label_cols:
            if col not in self.tasks:
                if col in self.secondary_structure_cols:
                    task = Task(TaskType.Binary, level=TaskLevel.Contact, num_labels=1)
                    self._tasks[col] = task  # type: ignore[index]
                    warn(
                        f"Secondary structure columns are assumed to be {task}."
                        " Please explicitly specify the task if this is not the case."
                    )
                else:
                    self._tasks[col] = self.infer_task(col, sequence_col)  # type: ignore[index]
        return self._tasks

    def infer_task(self, label_col: str, sequence_col: str | None = None) -> Task:
        if sequence_col is None:
            if len(self.sequence_cols) != 1:
                raise ValueError("sequence_col must be specified if there are multiple sequence columns.")
            sequence_col = self.sequence_cols[0]
        sequence = self._data.column(sequence_col)
        column = self._data.column(label_col)
        return infer_task(
            sequence,
            column,
            truncation=self.truncation,
            max_seq_length=self.max_seq_length,
            seq_length_offset=self.seq_length_offset,
        )

    def infer_discrete_map(self, discrete_map: Mapping | None = None):
        self._discrete_map = discrete_map or NestedDict()
        ignored_cols = set(self.discrete_map.keys()) | set(self.sequence_cols) | set(self.secondary_structure_cols)
        data_cols = [i for i in self.data_cols if i not in ignored_cols]
        for col in data_cols:
            discrete_map = infer_discrete_map(self._data.column(col))
            if discrete_map:
                self._discrete_map[col] = discrete_map  # type: ignore[index]
        return self._discrete_map

    def __getitems__(self, keys: int | slice | Iterable[int]) -> Any:
        return self.__getitem__(keys)

    def identify_special_cols(
        self, feature_cols: List | None = None, label_cols: List | None = None, id_cols: List | None = None
    ) -> Sequence:
        all_cols = self.data.column_names
        self._id_cols = id_cols or [i for i in all_cols if i in defaults.ID_COL_NAMES]

        string_cols = [k for k, v in self.features.items() if k not in self.id_cols and v.dtype == "string"]
        self._sequence_cols = [i for i in string_cols if i in defaults.SEQUENCE_COL_NAMES]
        self._secondary_structure_cols = [i for i in string_cols if i in defaults.SECONDARY_STRUCTURE_COL_NAMES]

        data_cols = [i for i in all_cols if i not in self.id_cols]
        if label_cols is None:
            if feature_cols is None:
                feature_cols = [i for i in data_cols if i in defaults.SEQUENCE_COL_NAMES]
            label_cols = [i for i in data_cols if i not in feature_cols]
        self._label_cols = label_cols
        if feature_cols is None:
            feature_cols = [i for i in data_cols if i not in self.label_cols]
        self._feature_cols = feature_cols
        missing_feature_cols = set(self.feature_cols).difference(data_cols)
        if missing_feature_cols:
            raise ValueError(f"{missing_feature_cols} are specified in feature_cols, but not found in dataset.")
        missing_label_cols = set(self.label_cols).difference(data_cols)
        if missing_label_cols:
            raise ValueError(f"{missing_label_cols} are specified in label_cols, but not found in dataset.")
        return string_cols

    def tokenize(self, string: str) -> Tensor:
        return self.tokenizer(string, return_attention_mask=False, truncation=self.truncation)["input_ids"]

    def tokenization(self, data: Mapping[str, str]) -> Mapping[str, Tensor]:
        return {col: self.tokenize(data[col]) for col in self.sequence_cols}

    def convert_secondary_structure(self, data: Mapping) -> Mapping:
        return {col: dot_bracket_to_contact_map(data[col]) for col in self.secondary_structure_cols}

    def map_discrete(self, data: Mapping) -> Mapping:
        return {name: map_value(data[name], mapping) for name, mapping in self.discrete_map.items()}

    def truncate(self, data: Mapping, columns: List[str], max_seq_length: int) -> Mapping:
        return {name: truncate_value(data[name], max_seq_length, self.tasks[name].level) for name in columns}

    def update(self, dataset: datasets.Dataset):
        r"""
        Perform an in-place update of the dataset.

        This method is used to update the dataset after changes have been made to the underlying data.
        It updates the format columns, data, info, and fingerprint of the dataset.
        """
        # pylint: disable=W0212
        # Why datasets won't support in-place changes?
        # It's just impossible to extend.
        self._format_columns = dataset._format_columns
        self._data = dataset._data
        self._info = dataset._info
        self._fingerprint = dataset._fingerprint

    def rename_columns(self, column_mapping: Mapping[str, str], new_fingerprint: str | None = None) -> datasets.Dataset:
        self.update(super().rename_columns(column_mapping, new_fingerprint=new_fingerprint))
        self._id_cols = [column_mapping.get(i, i) for i in self.id_cols]
        self._feature_cols = [column_mapping.get(i, i) for i in self.feature_cols]
        self._label_cols = [column_mapping.get(i, i) for i in self.label_cols]
        self._sequence_cols = [column_mapping.get(i, i) for i in self.sequence_cols]
        self._secondary_structure_cols = [column_mapping.get(i, i) for i in self.secondary_structure_cols]
        self._tasks = {column_mapping.get(k, k): v for k, v in self.tasks.items()}
        return self

    def rename_column(
        self, original_column_name: str, new_column_name: str, new_fingerprint: str | None = None
    ) -> datasets.Dataset:
        self.update(super().rename_column(original_column_name, new_column_name, new_fingerprint))
        self._id_cols = [new_column_name if i == original_column_name else i for i in self.id_cols]
        self._feature_cols = [new_column_name if i == original_column_name else i for i in self.feature_cols]
        self._label_cols = [new_column_name if i == original_column_name else i for i in self.label_cols]
        self._sequence_cols = [new_column_name if i == original_column_name else i for i in self.sequence_cols]
        self._secondary_structure_cols = [
            new_column_name if i == original_column_name else i for i in self.secondary_structure_cols
        ]
        self._tasks = {new_column_name if k == original_column_name else k: v for k, v in self.tasks.items()}
        return self

    def process_nan(self, data: Table, nan_process: str | None, fill_value: str | int | float = 0) -> Table:
        if nan_process == "ignore":
            return data
        data = data.to_pandas()
        data = data.replace([float("inf"), -float("inf")], float("nan"))
        if data.isnull().values.any():
            if nan_process is None or nan_process == "error":
                raise ValueError("NaN / inf values have been found in the dataset.")
            warn(
                "NaN / inf values have been found in the dataset.\n"
                "While we can handle them, the data type of the corresponding column may be set to float, "
                "which can and very likely will disrupt the auto task recognition.\n"
                "It is recommended to address these values before loading the dataset."
            )
            if nan_process == "drop":
                data = data.dropna()
            elif nan_process == "fill":
                data = data.fillna(fill_value)
            else:
                raise ValueError(f"Invalid nan_process: {nan_process}")
        return pa.Table.from_pandas(data)

    @property
    def id_cols(self) -> List:
        return self._id_cols

    @property
    def data_cols(self) -> List:
        return self.feature_cols + self.label_cols

    @property
    def feature_cols(self) -> List:
        return self._feature_cols

    @property
    def label_cols(self) -> List:
        return self._label_cols

    @property
    def sequence_cols(self) -> List:
        return self._sequence_cols

    @property
    def secondary_structure_cols(self) -> List:
        return self._secondary_structure_cols

    @property
    def tasks(self) -> NestedDict:
        if not hasattr(self, "_tasks"):
            return self.infer_tasks()
        return self._tasks

    @property
    def discrete_map(self) -> Mapping:
        if not hasattr(self, "_discrete_map"):
            return self.infer_discrete_map()
        return self._discrete_map
