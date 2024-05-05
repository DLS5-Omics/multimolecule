from __future__ import annotations

from typing import Tuple

from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from .generic import ClassificationHead
from .output import HeadOutput


class SequenceClassificationHead(ClassificationHead):
    """Head for sequence-level tasks."""

    def forward(
        self, outputs: ModelOutput | Tuple[Tensor, ...], labels: Tensor | None = None
    ) -> HeadOutput:  # pylint: disable=arguments-renamed
        return super().forward(outputs[1], labels)
