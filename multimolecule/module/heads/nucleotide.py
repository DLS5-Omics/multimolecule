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
from typing import TYPE_CHECKING, Mapping, Tuple

import torch
from chanfig import ConfigRegistry
from danling import NestedTensor
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from .config import HeadConfig
from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .utils import unfold_kmer_embeddings

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig

NucleotideHeadRegistryHF = ConfigRegistry(key="tokenizer_type")


@HeadRegistry.nucleotide.register("single", default=True)
@NucleotideHeadRegistryHF.register("single", default=True)
class NucleotidePredictionHead(PredictionHead):
    r"""
    Head for tasks in nucleotide-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "last_hidden_state"
    r"""The default output to use for the head."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the NucleotidePredictionHead.

        Args:
            outputs: The outputs of the model.
            attention_mask: The attention mask for the inputs.
            input_ids: The input ids for the inputs.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if attention_mask is None:
            if isinstance(input_ids, NestedTensor):
                input_ids, attention_mask = input_ids.tensor, input_ids.mask
            else:
                if input_ids is None:
                    raise ValueError(
                        f"Either attention_mask or input_ids must be provided for {self.__class__.__name__} to work."
                    )
                if self.pad_token_id is None:
                    raise ValueError(
                        f"pad_token_id must be provided when attention_mask is not passed to {self.__class__.__name__}."
                    )
                attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        output *= attention_mask.unsqueeze(-1)

        # remove cls token embeddings
        if self.bos_token_id is not None:
            output = output[..., 1:, :]
            # process attention_mask and input_ids to make removal of eos token happy
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token embeddings
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output)
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) == last_valid_indices.unsqueeze(1)
            output *= eos_mask[:, :, None]
            output = output[..., :-1, :]

        return super().forward(output, labels, **kwargs)


@HeadRegistry.register("nucleotide.kmer")
@NucleotideHeadRegistryHF.register("kmer")
class NucleotideKMerHead(PredictionHead):
    r"""
    Head for tasks in nucleotide-level with kmer inputs.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "last_hidden_state"
    r"""The default output to use for the head."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.nmers = config.nmers
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name
        # Do not pass bos_token_id and eos_token_id to unfold_kmer_embeddings
        # As they will be removed in preprocess
        self.unfold_kmer_embeddings = partial(unfold_kmer_embeddings, nmers=self.nmers)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the NucleotideKMerHead.

        Args:
            outputs: The outputs of the model.
            attention_mask: The attention mask for the inputs.
            input_ids: The input ids for the inputs.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if attention_mask is None:
            if isinstance(input_ids, NestedTensor):
                input_ids, attention_mask = input_ids.tensor, input_ids.mask
            else:
                if input_ids is None:
                    raise ValueError(
                        f"Either attention_mask or input_ids must be provided for {self.__class__.__name__} to work."
                    )
                if self.pad_token_id is None:
                    raise ValueError(
                        f"pad_token_id must be provided when attention_mask is not passed to {self.__class__.__name__}."
                    )
                attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        output = output * attention_mask.unsqueeze(-1)

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
        return super().forward(output, labels, **kwargs)
