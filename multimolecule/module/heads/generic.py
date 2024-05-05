from __future__ import annotations

from torch import Tensor, nn
from transformers.activations import ACT2FN

from multimolecule.models.configuration_utils import PretrainedConfig

from ..criterions import Criterion
from .output import HeadOutput
from .transform import HeadTransforms


class ClassificationHead(nn.Module):
    """Head for all-level of tasks."""

    num_labels: int

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config.head
        if self.config.hidden_size is None:
            self.config.hidden_size = config.hidden_size
        if self.config.num_labels is None:
            self.config.num_labels = config.num_labels
        if self.config.problem_type is None:
            self.config.problem_type = config.problem_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransforms.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = Criterion(self.config)

    def forward(self, embeddings: Tensor, labels: Tensor | None) -> HeadOutput:
        output = self.dropout(embeddings)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)
