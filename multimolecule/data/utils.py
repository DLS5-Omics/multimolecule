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

from typing import Any, Tuple

import pyarrow as pa
from pyarrow import Array, ChunkedArray, ListArray, StringArray

from multimolecule import defaults
from multimolecule.tasks import Task, TaskLevel, TaskType


def no_collate(batch: Any) -> Any:
    return batch


def infer_task(sequence: ChunkedArray | ListArray, column: Array | ChunkedArray | ListArray) -> Task:
    if isinstance(sequence, ChunkedArray) and len(sequence.chunks) == 1:
        sequence = sequence.chunks[0]
    if isinstance(column, ChunkedArray) and len(column.chunks) == 1:
        column = column.chunks[0]
    flattened, levels = flatten_column(column)
    dtype = flattened.type
    unique = flattened.unique()
    num_elem = len(sequence)
    num_tokens, num_contacts = get_num_tokens(sequence)

    if levels == 0 or (levels == 1 and len(flattened) % len(column) == 0):
        level = TaskLevel.Sequence
        num_labels = len(flattened) // num_elem
    else:
        num_rows = defaults.TASK_INFERENCE_NUM_ROWS
        sequence, column = sequence[:num_rows], column[:num_rows]
        num_tokens, num_contacts = get_num_tokens(sequence)
        if len(flattened) % num_contacts == 0:
            level = TaskLevel.Contact
            num_labels = len(flattened) // num_contacts
        elif len(flattened) % num_tokens == 0:
            level = TaskLevel.Token
            num_labels = len(flattened) // num_tokens
        else:
            raise ValueError("Unable to infer task: inconsistent number of values in sequence and column")

    if dtype in (pa.float16(), pa.float32(), pa.float64()):
        return Task(TaskType.Regression, level=level, num_labels=num_labels)
    if dtype in (pa.int8(), pa.int16(), pa.int32(), pa.int64()):
        if len(unique) == 2:
            if len(flattened) in (num_elem, num_tokens, num_contacts):
                return Task(TaskType.Binary, level=level, num_labels=1)
            return Task(TaskType.MultiLabel, level=level, num_labels=num_labels)
        if len(unique) / len(column) > defaults.LABLE_TYPE_THRESHOLD:
            return Task(TaskType.Regression, level=level, num_labels=num_labels)
        return Task(TaskType.MultiClass, level=level, num_labels=len(unique))
    raise ValueError(f"Unable to infer task: unsupported dtype {dtype}")


def flatten_column(column: Array | ChunkedArray | ListArray) -> Tuple[Array, int]:
    levels = 0
    while isinstance(column, (ChunkedArray, ListArray)):
        column = column.flatten()
        levels += 1
    return column, levels


def get_num_tokens(sequence: Array | ListArray) -> Tuple[int, int]:
    if isinstance(sequence, StringArray):
        return sum(len(i.as_py()) for i in sequence), sum(len(i.as_py()) ** 2 for i in sequence)
    # remove <bos> and <eos> tokens in length calculation
    offset = 0
    if len({i[0] for i in sequence}) == 1:
        offset += 1
    if len({i[-1] for i in sequence}) == 1:
        offset += 1
    return sum((len(i) - offset) for i in sequence), sum((len(i) - offset) ** 2 for i in sequence)
