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

from chanfig import ConfigRegistry as Registry_
from torch import nn


class Registry(Registry_):  # pylint: disable=too-few-public-methods
    key = "problem_type"

    def build(self, config) -> nn.Module:  # type: ignore[override]
        name = getattr(config, self.getattr("key"))
        return self.init(self.lookup(name), config)  # type: ignore[arg-type]


CriterionRegistry = Registry(fallback=True)
