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


from chanfig import Config

from multimolecule import MultiMoleculeConfig, MultiMoleculeRunner

# Sample 10% of the dataset to speed up testing process
data = Config(root="multimolecule/rivas-a", ratio=0.1)
config = MultiMoleculeConfig(data=data, pretrained="multimolecule/rnabert", epoch_end=1)
config.parse()
runner = MultiMoleculeRunner(config)

runner.train()
