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

from collections import OrderedDict
from dataclasses import dataclass


class BaseHeadConfig(OrderedDict):
    pass


@dataclass
class HeadConfig(BaseHeadConfig):
    r"""
    Configuration class for a prediction head.

    Args:
        num_labels:
            Number of labels to use in the last layer added to the model, typically for a classification task.

            Head should look for [`Config.num_labels`][multimolecule.PreTrainedConfig] if is `None`.
        problem_type:
            Problem type for `XxxForYyyPrediction` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.

            Head should look for [`Config.problem_type`][multimolecule.PreTrainedConfig] if is `None`.
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.

            Head should look for [`Config.hidden_size`][multimolecule.PreTrainedConfig] if is `None`.
        dropout:
            The dropout ratio for the hidden states.
        transform:
            The transform operation applied to hidden states.
        transform_act:
            The activation function of transform applied to hidden states.
        bias:
            Whether to apply bias to the final prediction layer.
        act:
            The activation function of the final prediction output.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        output_name (`str`, *optional*):
            The name of the tensor required in model outputs.

            If is `None`, will use the default output name of the corresponding head.
    """

    num_labels: int = None  # type: ignore[assignment]
    problem_type: str = None  # type: ignore[assignment]
    hidden_size: int | None = None
    dropout: float = 0.0
    transform: str | None = None
    transform_act: str | None = "gelu"
    bias: bool = True
    act: str | None = None
    layer_norm_eps: float = 1e-12
    output_name: str | None = None


@dataclass
class MaskedLMHeadConfig(BaseHeadConfig):
    r"""
    Configuration class for a Masked Language Modeling head.

    Args:
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.

            Head should look for [`Config.hidden_size`][multimolecule.PreTrainedConfig] if is `None`.
        dropout:
            The dropout ratio for the hidden states.
        transform:
            The transform operation applied to hidden states.
        transform_act:
            The activation function of transform applied to hidden states.
        bias:
            Whether to apply bias to the final prediction layer.
        act:
            The activation function of the final prediction output.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        output_name (`str`, *optional*):
            The name of the tensor required in model outputs.

            If is `None`, will use the default output name of the corresponding head.
    """

    hidden_size: int | None = None
    dropout: float = 0.0
    transform: str | None = "nonlinear"
    transform_act: str | None = "gelu"
    bias: bool = True
    act: str | None = None
    layer_norm_eps: float = 1e-12
    output_name: str | None = None
