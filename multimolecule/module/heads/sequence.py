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

from typing import TYPE_CHECKING, Mapping, Tuple

from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from .config import HeadConfig
from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


@HeadRegistry.register("sequence")
class SequencePredictionHead(PredictionHead):
    r"""
    Head for tasks in sequence-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "pooler_output"
    r"""The default output to use for the head."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the SequencePredictionHead.

        Args:
            outputs: The outputs of the model.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[1]
        return super().forward(output, labels, **kwargs)
