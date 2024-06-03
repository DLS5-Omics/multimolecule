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

import torch
from chanfig import ConfigRegistry
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from multimolecule.models.configuration_utils import HeadConfig, PreTrainedConfig

from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .utils import unfold_kmer_embeddings

NucleotideHeadRegistryHF = ConfigRegistry(key="tokenizer_type")


@HeadRegistry.register("nucleotide.single")
@NucleotideHeadRegistryHF.register("single", default=True)
class NucleotidePredictionHead(PredictionHead):
    """Head for nucleotide-level tasks."""

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
                raise ValueError(
                    "Either attention_mask or input_ids must be provided for NucleotidePredictionHead to work."
                )
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to NucleotidePredictionHead."
                )
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0] * attention_mask.unsqueeze(-1)
        # remove cls token embeddings
        if self.bos_token_id is not None:
            output = output[..., 1:, :]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token embeddings
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output)
                input_ids = input_ids[..., :-1]
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) == last_valid_indices.unsqueeze(1)
            output *= eos_mask[:, :, None]
            output = output[..., :-1, :]
            attention_mask = attention_mask[..., 1:]

        return super().forward(output, labels)


@HeadRegistry.register("nucleotide.kmer")
@NucleotideHeadRegistryHF.register("kmer")
class NucleotideKMerHead(PredictionHead):
    """Head for nucleotide-level tasks."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.nmers = config.nmers
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        # Do not pass bos_token_id and eos_token_id to unfold_kmer_embeddings
        # As they will be removed in preprocess
        self.unfold_kmer_embeddings = partial(unfold_kmer_embeddings, nmers=self.nmers)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided for NucleotideKMerHead to work.")
            if self.pad_token_id is None:
                raise ValueError(
                    "pad_token_id must be provided when attention_mask is not passed to NucleotideKMerHead."
                )
            attention_mask = input_ids.ne(self.pad_token_id)

        output = outputs[0]
        # remove cls token embeddings
        if self.bos_token_id is not None:
            output = output[..., 1:, :]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token embeddings
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output)
                input_ids = input_ids[..., :-1]
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) == last_valid_indices.unsqueeze(1)
            output *= eos_mask[:, :, None]
            output = output[..., :-1, :]
            attention_mask = attention_mask[..., 1:]

        output = self.unfold_kmer_embeddings(output, attention_mask)
        return super().forward(output, labels)
