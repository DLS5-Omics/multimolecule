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

import pyarrow as pa
import torch

from multimolecule.data.utils import infer_discrete_map, map_value, truncate_batch, truncate_value
from multimolecule.tasks import TaskLevel


class TestTruncateValue:
    def test_contact_tensor(self):
        value = torch.arange(25).reshape(5, 5)

        output = truncate_value(value, 3, TaskLevel.Contact)

        torch.testing.assert_close(output, value[:3, :3])


class TestTruncateBatch:
    def test_token_batch(self):
        batch = [torch.arange(5), torch.arange(7)]

        output = truncate_batch(batch, 3, TaskLevel.Token)

        assert [tuple(item.shape) for item in output] == [(3,), (3,)]

    def test_contact_batch(self):
        batch = [torch.arange(25).reshape(5, 5), torch.arange(49).reshape(7, 7)]

        output = truncate_batch(batch, 3, TaskLevel.Contact)

        assert [tuple(item.shape) for item in output] == [(3, 3), (3, 3)]


class TestInferDiscreteMap:
    def test_string_column_skips_nulls(self):
        column = pa.array(["abc", None, "ba"])

        assert infer_discrete_map(column) == {"a": 0, "b": 1, "c": 2}


class TestMapValue:
    def test_nested_strings_preserve_nulls(self):
        mapping = {"a": 0, "b": 1, "c": 2}

        assert map_value(["abc", None, "ba"], mapping) == [[0, 1, 2], None, [1, 0]]
