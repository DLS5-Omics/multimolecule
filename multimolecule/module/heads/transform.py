from __future__ import annotations

from chanfig import ConfigRegistry
from torch import Tensor, nn
from transformers.activations import ACT2FN

from multimolecule.models.configuration_utils import BaseHeadConfig

HeadTransforms = ConfigRegistry(key="transform")


@HeadTransforms.register("nonlinear")
class NonLinearTransform(nn.Module):
    def __init__(self, config: BaseHeadConfig):
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


@HeadTransforms.register("linear")
class LinearTransform(nn.Module):
    def __init__(self, config: BaseHeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@HeadTransforms.register(None)
class IdentityTransform(nn.Identity):
    def __init__(self, config: BaseHeadConfig):  # pylint: disable=unused-argument
        super().__init__()
