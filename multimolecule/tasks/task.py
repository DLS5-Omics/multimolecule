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

from dataclasses import dataclass
from enum import Enum

from chanfig import NestedDict


class TaskType(str, Enum):
    Binary = "binary"
    MultiClass = "multiclass"
    MultiLabel = "multilabel"
    Regression = "regression"


class TaskLevel(str, Enum):
    Sequence = "sequence"
    Token = "token"
    Contact = "contact"


@dataclass
class Task(NestedDict):
    type: TaskType
    level: TaskLevel
    num_labels: int = 1

    def __post_init__(self):
        if self.type in (TaskType.Binary) and self.num_labels != 1:
            raise ValueError(f"num_labels must be 1 for {self.type} task")
        if self.type in (TaskType.MultiClass, TaskType.MultiLabel) and self.num_labels == 1:
            raise ValueError(f"num_labels must not be 1 for {self.type} task")
        super().__post_init__()
