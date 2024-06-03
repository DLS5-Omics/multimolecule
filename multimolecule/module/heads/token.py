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

from functools import partial
from typing import Tuple

from chanfig import ConfigRegistry
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from multimolecule.models.configuration_utils import HeadConfig, PreTrainedConfig

from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .utils import unfold_kmer_embeddings

TokenHeadRegistryHF = ConfigRegistry(key="tokenizer_type")


@HeadRegistry.register("token.single")
@TokenHeadRegistryHF.register("single", default=True)
class TokenPredictionHead(PredictionHead):
    """Head for token-level tasks."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided for TokenPredictionHead to work.")
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to TokenPredictionHead."
                )
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0] * attention_mask.unsqueeze(-1)
        return super().forward(output, labels)


@HeadRegistry.register("token.kmer")
@TokenHeadRegistryHF.register("kmer")
class TokenKMerHead(PredictionHead):
    """Head for token-level tasks."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
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

        output = self.unfold_kmer_embeddings(outputs[0], attention_mask)
        return super().forward(output, labels)
