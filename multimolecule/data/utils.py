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

from collections.abc import Iterable
from typing import Any, Tuple
from warnings import warn

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import Array, ChunkedArray, ListArray, StringArray

from multimolecule import defaults
from multimolecule.tasks import Task, TaskLevel, TaskType


def no_collate(batch: Any) -> Any:
    return batch


def infer_task(
    sequence: ChunkedArray | ListArray,
    column: Array | ChunkedArray | ListArray,
    *,
    truncation: bool = False,
    max_seq_length: int | None = None,
    seq_length_offset: int | None = None,
) -> Task:
    if max_seq_length is not None and seq_length_offset is not None:
        max_seq_length -= seq_length_offset
    if isinstance(sequence, ChunkedArray) and sequence.num_chunks == 1:
        sequence = sequence.chunks[0]
    if isinstance(column, ChunkedArray) and column.num_chunks == 1:
        column = column.chunks[0]
    flattened, levels = flatten_column(column, truncation, max_seq_length)
    dtype = flattened.type
    unique = flattened.unique()
    num_elem = len(sequence)
    num_tokens, num_contacts = get_num_tokens(sequence, seq_length_offset)

    if levels == 0 and len(sequence) == len(column):
        level = TaskLevel.Sequence
        num_labels = len(flattened) // num_elem
    elif levels > 0:
        if len(flattened) % num_contacts == 0:
            level = TaskLevel.Contact
            num_labels = len(flattened) // num_contacts
        elif len(flattened) % num_tokens == 0:
            level = TaskLevel.Nucleotide
            num_labels = len(flattened) // num_tokens
        elif len(flattened) % num_elem == 0:
            level = TaskLevel.Sequence
            num_labels = len(flattened) // num_elem
        else:
            raise ValueError("Unable to infer task: unsupported column structure")
    else:
        raise ValueError("Unable to infer task: unsupported column structure")

    if pa.types.is_floating(dtype):
        return Task(TaskType.Regression, level=level, num_labels=num_labels)
    if pa.types.is_integer(dtype):
        if len(unique) == 2:
            if len(flattened) in (num_elem, num_tokens, num_contacts):
                return Task(TaskType.Binary, level=level, num_labels=1)
            return Task(TaskType.MultiLabel, level=level, num_labels=num_labels)
        if len(unique) / len(column) > defaults.LABLE_TYPE_THRESHOLD:
            return Task(TaskType.Regression, level=level, num_labels=num_labels)
        return Task(TaskType.MultiClass, level=level, num_labels=len(unique))
    if pa.types.is_string(dtype):
        num_tokens_flattened, num_contacts_flattened = get_num_tokens(flattened)
        num_labels = len(set("".join(pa.array(flattened).to_pylist())))
        task_type = TaskType.MultiClass if num_labels > 2 else TaskType.Binary
        num_labels = 1 if task_type == TaskType.Binary else num_labels
        if num_tokens_flattened == num_tokens:
            return Task(task_type, level=TaskLevel.Nucleotide, num_labels=num_labels)
        if num_contacts_flattened == num_contacts:
            return Task(task_type, level=TaskLevel.Contact, num_labels=num_labels)
        return Task(task_type, level=TaskLevel.Sequence, num_labels=num_labels)
    raise ValueError(f"Unable to infer task: unsupported dtype {dtype}")


def infer_discrete_map(column: Array | ChunkedArray | ListArray) -> dict[str, int] | None:
    if pa.types.is_floating(column.type):
        return None
    flattened, _ = flatten_column(column)
    if pa.types.is_floating(flattened.type):
        return None
    if isinstance(flattened, (ChunkedArray, ListArray, StringArray)):
        unique = set()
        for i in flattened:
            unique.update(i.as_py())
    else:
        unique = flattened.unique().to_pylist()
    ret = {j: i for i, j in enumerate(sorted(unique))}
    if list(ret.keys()) == list(ret.values()):
        return None
    return ret


def map_value(value: Any, mapping: dict[str, int] | None) -> Any:
    if mapping is None:
        return value
    if isinstance(value, list) and isinstance(value[0], Iterable):
        return [[mapping[i] for i in j] for j in value]
    if isinstance(value, Iterable):
        return [mapping[i] for i in value]
    return mapping[value]


def truncate_value(value: Any, max_seq_length: int, level: int | None = None) -> Any:
    if level == TaskLevel.Nucleotide:
        return value[:max_seq_length]
    if level == TaskLevel.Contact:
        return [i[:max_seq_length] for i in value[:max_seq_length]]
    return value


def flatten_column(
    column: Array | ChunkedArray | ListArray, truncation: bool = False, max_seq_length: int | None = None
) -> Tuple[Array, int]:
    levels = 0
    while isinstance(column, (ChunkedArray, ListArray)):
        if isinstance(column, ChunkedArray):
            column = column.combine_chunks()
        elif isinstance(column, ListArray):
            if truncation and max_seq_length is not None and 0 < max_seq_length < 2**32:
                column = pc.list_slice(column, 0, max_seq_length)
            column = column.flatten()
            levels += 1
    return column, levels


def get_num_tokens(sequence: Array | ListArray, seq_length_offset: int | None = None) -> Tuple[int, int]:
    if isinstance(sequence, StringArray):
        return sum(len(i.as_py()) for i in sequence), sum(len(i.as_py()) ** 2 for i in sequence)
    # remove <bos> and <eos> tokens in length calculation
    if seq_length_offset is None:
        warn("seq_length_offset not specified, automatically detecting <bos> and <eos> tokens")
        seq_length_offset = 0
        if isinstance(sequence[0], pa.lib.StringScalar):
            raise ValueError("seq_length_offset must be specified for StringScalar sequences")
        if len({i[0] for i in sequence}) == 1:
            seq_length_offset += 1
        if len({i[-1] for i in sequence}) == 1:
            seq_length_offset += 1
    return compute_sums(sequence, seq_length_offset)


def compute_sums(sequence, seq_length_offset):
    total_sum, total_squared_sum = 0, 0
    for i in sequence:
        length_adjusted = get_length(i) - seq_length_offset
        total_sum += length_adjusted
        total_squared_sum += length_adjusted**2
    return total_sum, total_squared_sum


def get_length(string: str | pa.StringScalar):
    if isinstance(string, pa.StringScalar):
        return pc.utf8_length(string).as_py()
    return len(string)
