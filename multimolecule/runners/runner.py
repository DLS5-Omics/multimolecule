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

import danling as dl

from .base_runner import BaseRunner


class MultiMoleculeRunner(type):
    def __new__(cls, config):
        if config.get("platform", "torch") == "torch":
            return TorchRunner(config)
        if config.platform == "deepspeed":
            return DeepSpeedRunner(config)
        if config.platform == "accelerate":
            return AccelerateRunner(config)
        raise ValueError(f"Unsupported platform: {config.platform}")


class TorchRunner(BaseRunner, dl.TorchRunner):
    pass


class DeepSpeedRunner(BaseRunner, dl.DeepSpeedRunner):
    pass


class AccelerateRunner(BaseRunner, dl.AccelerateRunner):
    pass
