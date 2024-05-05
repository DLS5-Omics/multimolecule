from __future__ import annotations

from dataclasses import dataclass

from torch import FloatTensor
from transformers.modeling_outputs import ModelOutput


@dataclass
class HeadOutput(ModelOutput):
    """Output of a head."""

    logits: FloatTensor
    loss: FloatTensor | None = None
