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


from chanfig import ConfigRegistry as Registry_
from torch import nn


class Registry(Registry_):  # pylint: disable=too-few-public-methods
    """Registry for criterion classes that are selected based on problem_type.

    This registry extends the ConfigRegistry to provide a specialized mechanism
    for looking up and initializing criterion classes based on problem type.
    """

    key = "criterion"

    def build(self, config) -> nn.Module:  # type: ignore[override]
        """Build a criterion instance based on the problem_type in the config.

        Args:
            config: Configuration object containing problem_type and other
                   parameters needed to initialize the criterion.

        Returns:
            nn.Module: An initialized criterion instance.
        """
        name = getattr(config, self.getattr("key"), None)
        if name is None:
            name = config.problem_type
        return self.init(self.lookup(name), config)  # type: ignore[arg-type]


CRITERIONS = Registry(fallback=True)
