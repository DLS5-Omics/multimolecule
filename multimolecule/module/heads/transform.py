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

from chanfig import ConfigRegistry, Registry
from torch import Tensor, nn
from transformers.activations import ACT2FN

from multimolecule.models.configuration_utils import HeadConfig

HeadTransformRegistry = Registry()
HeadTransformRegistryHF = ConfigRegistry(key="transform")


@HeadTransformRegistry.register("nonlinear")
@HeadTransformRegistryHF.register("nonlinear")
class NonLinearTransform(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.transform_act, str):
            self.transform_act_fn = ACT2FN[config.transform_act]
        else:
            self.transform_act_fn = config.transform_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@HeadTransformRegistry.register("linear")
@HeadTransformRegistryHF.register("linear")
class LinearTransform(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@HeadTransformRegistry.register(None)
@HeadTransformRegistryHF.register(None)
class IdentityTransform(nn.Identity):
    def __init__(self, config: HeadConfig):  # pylint: disable=unused-argument
        super().__init__()
