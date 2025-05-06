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

from chanfig import ConfigRegistry, Registry
from torch import Tensor, nn
from transformers.activations import ACT2FN

from .config import HeadConfig

HEAD_TRANSFORMS = Registry()
HEAD_TRANSFORMS_HF = ConfigRegistry(key="transform")


@HEAD_TRANSFORMS.register("nonlinear")
@HEAD_TRANSFORMS_HF.register("nonlinear")
class NonLinearTransform(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.transform_act, str):
            self.activation = ACT2FN[config.transform_act]
        else:
            self.activation = config.transform_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@HEAD_TRANSFORMS.register("linear")
@HEAD_TRANSFORMS_HF.register("linear")
class LinearTransform(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@HEAD_TRANSFORMS.register("identity", default=True)
@HEAD_TRANSFORMS_HF.register("identity", default=True)
class IdentityTransform(nn.Identity):
    def __init__(self, config: HeadConfig):  # pylint: disable=unused-argument
        super().__init__()
