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

from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Any, List, Tuple
from warnings import warn

import danling as dl
import datasets
import pandas as pd
import pyarrow as pa
import torch
from chanfig import NestedDict
from danling import NestedTensor
from datasets import get_dataset_split_names
from datasets.table import Table
from numpy import random
from packaging.version import parse as parse_version
from pandas import DataFrame
from torch import Tensor
from torch.utils import data
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from multimolecule import defaults
from multimolecule.tasks import Task, TaskLevel, TaskType
from multimolecule.tokenisers.dna.utils import NUCLEOBASE_ALPHABET as DNA_MINIMAL_ALPHABET
from multimolecule.tokenisers.dna.utils import STANDARD_ALPHABET as DNA_COMPLETE_ALPHABET
from multimolecule.tokenisers.dot_bracket.utils import DOT_BRACKET_ALPHABET as DB_MINIMAL_ALPHABET
from multimolecule.tokenisers.dot_bracket.utils import STANDARD_ALPHABET as DB_COMPLETE_ALPHABET
from multimolecule.tokenisers.protein.utils import AMINO_ACID_ALPHABET as PROTEIN_MINIMAL_ALPHABET
from multimolecule.tokenisers.protein.utils import STANDARD_ALPHABET as PROTEIN_COMPLETE_ALPHABET
from multimolecule.tokenisers.rna.utils import NUCLEOBASE_ALPHABET as RNA_MINIMAL_ALPHABET
from multimolecule.tokenisers.rna.utils import STANDARD_ALPHABET as RNA_COMPLETE_ALPHABET
from multimolecule.utils.rna_secondary_structure import dot_bracket_to_contact_map

from .registry import DATASETS
from .utils import flatten_column, infer_discrete_map, infer_task, map_value, truncate_value

alphabets = {
    "dna": {
        "complete": DNA_COMPLETE_ALPHABET,
        "minimal": DNA_MINIMAL_ALPHABET,
    },
    "rna": {
        "complete": RNA_COMPLETE_ALPHABET,
        "minimal": RNA_MINIMAL_ALPHABET,
    },
    "protein": {
        "complete": PROTEIN_COMPLETE_ALPHABET,
        "minimal": PROTEIN_MINIMAL_ALPHABET,
    },
    "na": {
        "complete": DNA_COMPLETE_ALPHABET + RNA_COMPLETE_ALPHABET,
        "minimal": DNA_MINIMAL_ALPHABET + RNA_MINIMAL_ALPHABET,
    },
}

datasets.disable_progress_bars()


@DATASETS.register("auto", default=True)
class Dataset(datasets.Dataset):
    r"""
    The base class for all datasets.

    Dataset is a subclass of [`datasets.Dataset`][] that provides additional functionality for handling structured data.
    It has three main features:

    - column identification: identify the special columns (sequence and structure columns) in the dataset.
    - tokenization: tokenize the sequence columns in the dataset using a pretrained tokenizer.
    - task inference: infer the task type and level of each label column in the dataset.

    Attributes:
        tasks: A nested dictionary of [`Task`][]s for each label column in the dataset.
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
        auto_rename_sequence_col: Whether to automatically rename sequence columns to standard name.
            Only works when there is exactly one sequence column
            You can control the naming through `multimolecule.defaults.SEQUENCE_COL_NAME`.
            For more refined control, use `column_names_map`.
        auto_rename_label_col: Whether to automatically rename label column to standard name.
            Only works when there is exactly one label column.
            You can control the naming through `multimolecule.defaults.LABEL_COL_NAME`.
            For more refined control, use `column_names_map`.
        column_names_map: A mapping of column names to new column names.
            This is useful for renaming columns to inputs that are expected by a model.
            Defaults to `None`.
        truncation: Whether to truncate sequences that exceed the maximum length of the tokenizer.
            Defaults to `False`.
        max_seq_length: The maximum length of the input sequences.
            Defaults to the `model_max_length` of the tokenizer.
        tasks: A mapping of [`Tasks`][]s.
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

    _id_cols: List
    _feature_cols: List
    _label_cols: List

    _sequence_cols: List
    _secondary_structure_cols: List
    _sequence_type: str

    _tasks: NestedDict[str, Task]
    _discrete_map: Mapping

    tokenizer: PreTrainedTokenizerBase
    truncation: bool = True
    max_seq_length: int
    seq_length_offset: int = 0

    preprocess: bool = False
    auto_rename_sequence_col: bool = True
    auto_rename_label_col: bool = False
    column_names_map: Mapping[str, str] | None = None
    ignored_cols: List[str] = []

    def __init__(
        self,
        data: Table | DataFrame | dict | list | str,
        split: datasets.NamedSplit | None = None,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        id_cols: List | None = None,
        sequence_cols: List | None = None,
        ignored_cols: List | None = None,
        auto_rename_sequence_col: bool | None = None,
        auto_rename_label_col: bool | None = None,
        column_names_map: Mapping[str, str] | None = None,
        tasks: Mapping[str, Task] | None = None,
        discrete_map: Mapping[str, int] | None = None,
        sequence_type: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        pretrained: str | None = None,
        truncation: bool | None = None,
        max_seq_length: int | None = None,
        nan_process: str = "ignore",
        fill_value: str | int | float = 0,
        info: datasets.DatasetInfo | None = None,
        indices: Sequence | Table | None = None,
        fingerprint: str | None = None,
        preprocess: bool | None = None,
        train: bool | None = None,
    ):
        arrow_table, split = self.build_table(
            data, split, feature_cols, label_cols, nan_process=nan_process, fill_value=fill_value
        )
        if indices is not None and not isinstance(indices, (Table, pa.Table)):
            indices = pa.Table.from_arrays([pa.array(indices, type=pa.uint32())], names=["index"])
        super().__init__(
            arrow_table=arrow_table, split=split, info=info, indices_table=indices, fingerprint=fingerprint
        )
        self.post_init(
            feature_cols=feature_cols,
            label_cols=label_cols,
            id_cols=id_cols,
            sequence_cols=sequence_cols,
            ignored_cols=ignored_cols,
            tasks=tasks,
            discrete_map=discrete_map,
            auto_rename_sequence_col=auto_rename_sequence_col,
            auto_rename_label_col=auto_rename_label_col,
            column_names_map=column_names_map,
            sequence_type=sequence_type,
            tokenizer=tokenizer,
            pretrained=pretrained,
            truncation=truncation,
            max_seq_length=max_seq_length,
            preprocess=preprocess,
            train=train,
        )

    def build_table(
        self,
        data: Table | DataFrame | dict | str,
        split: datasets.NamedSplit | None = None,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        nan_process: str | None = "ignore",
        fill_value: str | int | float = 0,
    ) -> datasets.table.Table:
        splits = ()
        is_hf_dataset = False
        if isinstance(data, str):
            with suppress(FileNotFoundError):
                splits = get_dataset_split_names(data)
                is_hf_dataset = bool(splits)
        if is_hf_dataset:
            data, split = self._build_table_from_datasets(data, split, splits)
        else:
            data, split = self._build_table_from_local_dataset(data, split)
        if feature_cols is not None and label_cols is not None:
            data = data.select(feature_cols + label_cols)
        data = self.process_nan(data, nan_process=nan_process, fill_value=fill_value)
        return data, split

    def _build_table_from_datasets(
        self,
        data: Table | DataFrame | dict | str,
        split: datasets.NamedSplit | None = None,
        splits: Sequence[datasets.NamedSplit] | None = None,
    ) -> Tuple[datasets.table.Table, datasets.NamedSplit]:
        if not splits:
            splits = get_dataset_split_names(data)
        if split is None:
            if len(splits) != 1:  # type: ignore[arg-type]
                raise ValueError(f"Found multiple splits in the dataset: {splits}, but split is not specified.")
            split = splits[0]  # type: ignore[index]
        if split not in splits:  # type: ignore[operator]
            raise ValueError(f"Found the following splits in the dataset: {splits}, but {split} is not one of them.")
        return datasets.load_dataset(data, split=split).data, split

    def _build_table_from_local_dataset(
        self,
        data: Table | DataFrame | dict | str,
        split: datasets.NamedSplit | None = None,
    ) -> Tuple[datasets.table.Table, datasets.NamedSplit]:
        if isinstance(data, str):
            data = dl.load_pandas(data)
        if isinstance(data, dict):
            return pa.Table.from_pydict(data), split
        if isinstance(data, list):
            return pa.Table.from_pylist(data), split
        if isinstance(data, DataFrame):
            data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
            # If there are None values in the dataset, we replace them with nan and process them later at once.
            data = data.fillna(float("nan"))
            pd_version = parse_version(pd.__version__)
            if pd_version >= parse_version("2.1.0"):
                data = data.map(lambda x: [float("nan") if i is None else i for i in x] if isinstance(x, list) else x)
            else:
                data = data.applymap(
                    lambda x: [float("nan") if i is None else i for i in x] if isinstance(x, list) else x
                )
            return pa.Table.from_pandas(data, preserve_index=False), split
        raise ValueError(f"Unsupported data type: {type(data)}")

    def post_init(
        self,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        id_cols: List | None = None,
        sequence_cols: List | None = None,
        ignored_cols: List | None = None,
        tasks: Mapping[str, Task] | None = None,
        discrete_map: Mapping[str, int] | None = None,
        auto_rename_sequence_col: bool | None = None,
        auto_rename_label_col: bool | None = None,
        column_names_map: Mapping[str, str] | None = None,
        sequence_type: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        pretrained: str | None = None,
        max_seq_length: int | None = None,
        truncation: bool | None = None,
        preprocess: bool | None = None,
        train: bool | None = None,
    ) -> None:
        r"""
        Perform pre-processing steps after initialization.

        It first identifies the special columns (sequence and structure columns) in the dataset.
        Then it sets the feature and label columns based on the input arguments.
        If `auto_rename_sequence_col` is `True`, it will automatically rename the sequence column.
        If `auto_rename_label_col` is `True`, it will automatically rename the label column.
        Finally, it sets the [`transform`][datasets.Dataset.set_transform] function based on the `preprocess` flag.
        """

        # Process columns
        self.identify_special_cols(
            feature_cols=feature_cols,
            label_cols=label_cols,
            id_cols=id_cols,
            sequence_cols=sequence_cols,
            sequence_type=sequence_type,
        )
        self.ignored_cols = ignored_cols or self.id_cols
        if auto_rename_sequence_col is not None:
            self.auto_rename_sequence_col = auto_rename_sequence_col
        if auto_rename_label_col is not None:
            self.auto_rename_label_col = auto_rename_label_col
        if column_names_map is None:
            column_names_map = {}
        if self.auto_rename_sequence_col:
            if len(self.sequence_cols) != 1:
                raise ValueError("auto_rename_sequence_col can only be used when there is exactly one sequence column.")
            column_names_map[self.sequence_cols[0]] = defaults.SEQUENCE_COL_NAME  # type: ignore[index]
        if self.auto_rename_label_col:
            if len(self.label_cols) != 1:
                raise ValueError("auto_rename_label_col can only be used when there is exactly one label column.")
            column_names_map[self.label_cols[0]] = defaults.LABEL_COL_NAME  # type: ignore[index]
        self.column_names_map = column_names_map
        if self.column_names_map:
            self.rename_columns(self.column_names_map)

        # Initialize tokenizer
        if tokenizer is None:
            if not pretrained:
                if not self.sequence_type:
                    raise ValueError("Either pretrained or sequence_type must be specified.")
                pretrained = "multimolecule/" + self.sequence_type.lower()
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

        # Infer Tasks & Discrete Map
        if tasks is not None:
            self.tasks = tasks
        else:
            self.infer_tasks()
        if discrete_map is not None:
            self._discrete_map = discrete_map
        if train is None:
            if self.split is not None:
                train = self.split.lower() in defaults.TRAIN_SPLITS
            else:
                warn("Neither split nor train is specified. Assuming the dataset is for evaluation.")
                train = False
        self.train = train

        # Preprocess
        if preprocess is not None:
            self.preprocess = preprocess
        if self.preprocess:
            self.update(self.map(self.tokenization))
            if self.secondary_structure_cols:
                self.update(self.map(self.convert_secondary_structure))
            if self.discrete_map:
                self.update(self.map(self.map_discrete))
            fn_kwargs = {
                "columns": [name for name, task in self.tasks.items() if task.level in ["token", "contact"]],
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
            if col in self.secondary_structure_cols:
                data = [torch.tensor(dot_bracket_to_contact_map(d)) for d in data]
            elif col in self.discrete_map:
                data = map_value(data, self.discrete_map[col])
            if col in self.tasks:
                data = truncate_value(data, self.max_seq_length - self.seq_length_offset, self.tasks[col].level)
        if col in self.tasks:
            ignore_value = float("nan") if self.tasks[col].type == TaskType.Regression else -100
            if isinstance(data[0], list):
                data = [[i if i is not None else ignore_value for i in d] for d in data]
            else:
                data = [i if i is not None else ignore_value for i in data]
        if isinstance(data[0], str):
            return data
        try:
            return torch.tensor(data)
        except (TypeError, ValueError):
            return NestedTensor(data)

    def infer_tasks(self, sequence_col: str | None = None) -> NestedDict:
        for col in self.label_cols:
            if col in self.tasks:
                continue
            try:
                self.tasks[col] = self.infer_task(col, sequence_col)  # type: ignore[index]
            except ValueError:
                raise ValueError(f"Unable to infer task for column {col}.")
        return self.tasks

    def infer_task(self, label_col: str, sequence_col: str | None = None) -> Task:
        if sequence_col is None:
            if len(self.sequence_cols) != 1:
                raise ValueError("sequence_col must be specified if there are multiple sequence columns.")
            sequence_col = self.sequence_cols[0]
        if label_col in self.secondary_structure_cols:
            task = Task(TaskType.Binary, level=TaskLevel.Contact, num_labels=1)
            warn(
                f"Secondary structure columns are assumed to be {task}. "
                "Please explicitly specify the task if this is not the case."
            )
            return task
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
        self,
        feature_cols: List | None = None,
        label_cols: List | None = None,
        id_cols: List | None = None,
        sequence_cols: List | None = None,
        sequence_type: str | None = None,
    ) -> Sequence:
        all_cols = self.data.column_names
        self._id_cols = id_cols or [i for i in all_cols if i.lower() in defaults.ID_COL_NAMES]
        self._sequence_cols = sequence_cols or []
        self._sequence_type = sequence_type  # type: ignore[assignment]

        string_cols = [k for k, v in self.features.items() if k not in self.id_cols and v.pa_type == "string"]
        unique_chars = {
            k: {ch for s in flatten_column(self._data.column(k))[0] for ch in s.as_py()} for k in string_cols
        }
        unique_chars_upper = {k: {ch.upper() for ch in v} for k, v in unique_chars.items()}

        for col, chars in unique_chars_upper.items():
            for alphabet_type, alphabet in alphabets.items():
                complete, minimal = alphabet["complete"], alphabet["minimal"]
                if chars.issubset(complete) and chars.issuperset(minimal):
                    self._sequence_cols.append(col)
                    if self._sequence_type is None and alphabet_type != "na":
                        self._sequence_type = alphabet_type

        if not self._sequence_cols:
            raise ValueError("No sequence column found in the dataset.")

        self._secondary_structure_cols = [
            k for k, v in unique_chars.items() if v.issubset(DB_COMPLETE_ALPHABET) and v.issuperset(DB_MINIMAL_ALPHABET)
        ]

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
        if getattr(self, "_tasks", None) is not None:
            self.tasks = {column_mapping.get(k, k): v for k, v in self.tasks.items()}
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
        if getattr(self, "_tasks", None) is not None:
            self.tasks = {new_column_name if k == original_column_name else k: v for k, v in self.tasks.items()}
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
        return pa.Table.from_pandas(data, preserve_index=False)

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
    def sequence_type(self) -> str:
        return self._sequence_type

    @property
    def tasks(self) -> NestedDict:
        if not hasattr(self, "_tasks"):
            self._tasks = NestedDict()
            return self.infer_tasks()
        return self._tasks

    @tasks.setter
    def tasks(self, tasks: Mapping):
        self._tasks = NestedDict()
        for name, task in tasks.items():
            if not isinstance(task, Task):
                task = Task(**task)
            self._tasks[name] = task

    @property
    def discrete_map(self) -> Mapping:
        if not hasattr(self, "_discrete_map"):
            return self.infer_discrete_map()
        return self._discrete_map


@DATASETS.register("sample")
class SampleDataset(data.Dataset):

    dataset: Dataset
    ratio: float | int = 1
    indices: Sequence

    def __init__(self, dataset: Dataset, ratio: float | int = 1, seed: int = 0):
        if ratio <= 0:
            raise ValueError(f"Invalid ratio: {ratio}")
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed
        self._epoch = 0
        self._access_count = 0
        self.indices = self._sample_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        if isinstance(index, (Sequence, slice)):
            return self.__getitems__(index)
        elif not isinstance(index, int):
            raise ValueError(f"Invalid index type: {type(index)}")
        index = self.indices[index]
        self._access_count += 1
        if self._access_count >= len(self):
            self.epoch += 1
        return self.dataset[index]

    def __getitems__(self, indices: Sequence | slice):
        if isinstance(indices, int):
            return self.__getitem__(indices)
        if isinstance(indices, Sequence):
            indices = [self.indices[i] for i in indices]
        elif isinstance(indices, slice):
            indices = self.indices[indices]
        else:
            raise ValueError(f"Invalid index type: {type(indices)}")
        self._access_count += len(indices)
        if self._access_count >= len(self):
            self.epoch += 1
        if hasattr(self.dataset, "__getitems__"):
            return self.dataset.__getitems__(indices)
        return [self.dataset[i] for i in indices]

    def _sample_indices(self):
        g = random.default_rng(self.seed + self.epoch)
        integer = int(self.ratio)
        decimal = self.ratio - integer
        if decimal == 0:
            return [i for i in range(len(self.dataset)) for _ in range(integer)]
        if integer == 0:
            return sorted(g.choice(range(len(self.dataset)), size=int(len(self.dataset) * decimal), replace=False))
        indices = [i for i in range(len(self.dataset)) for _ in range(integer)]
        indices.extend(g.choice(range(len(self.dataset)), size=int(len(self.dataset) * decimal), replace=False))
        return sorted(indices)

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'dataset'. "
                "If this instance is initialized, this is probably because of multiprocessing."
            )
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, ratio={self.ratio}, seed={self.seed})"

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.indices = self._sample_indices()
        self._access_count = 0
