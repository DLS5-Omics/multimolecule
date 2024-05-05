from __future__ import annotations

from functools import partial
from typing import Tuple

from chanfig import ConfigRegistry
from torch import Tensor
from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from .generic import ClassificationHead
from .output import HeadOutput
from .utils import unfold_kmer_embeddings

TokenHeads = ConfigRegistry(key="tokenizer_type")


@TokenHeads.register("single", default=True)
class TokenClassificationHead(ClassificationHead):
    """Head for token-level tasks."""

    def forward(
        self, outputs: ModelOutput | Tuple[Tensor, ...], labels: Tensor | None = None
    ) -> HeadOutput:  # pylint: disable=arguments-renamed
        return super().forward(outputs[0], labels)


@TokenHeads.register("kmer")
class TokenKMerHead(ClassificationHead):
    """Head for token-level tasks."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.nmers = config.nmers
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.unfold_kmer_embeddings = partial(
            unfold_kmer_embeddings, nmers=self.nmers, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
        )

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided for TokenKMerHead to work.")
            if self.pad_token_id is None:
                raise ValueError("pad_token_id must be provided when attention_mask is not passed to TokenKMerHead.")
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0]
        output = self.unfold_kmer_embeddings(output, attention_mask)
        return super().forward(output, labels)
