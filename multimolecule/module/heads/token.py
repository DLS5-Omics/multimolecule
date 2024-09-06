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

TokenHeadRegistryHF = ConfigRegistry(key="tokenizer_type")


@HeadRegistry.token.register("single", default=True)
@TokenHeadRegistryHF.register("single", default=True)
class TokenPredictionHead(PredictionHead):
    r"""
    Head for tasks in token-level.

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
        Forward pass of the TokenPredictionHead.

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
        return super().forward(output, labels, **kwargs)


@HeadRegistry.register("token.kmer")
@TokenHeadRegistryHF.register("kmer")
class TokenKMerHead(PredictionHead):
    r"""
    Head for tasks in token-level with kmer inputs.

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
        self.unfold_kmer_embeddings = partial(
            unfold_kmer_embeddings, nmers=self.nmers, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
        )

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
        Forward pass of the TokenKMerHead.

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
        output = self.unfold_kmer_embeddings(output, attention_mask)
        return super().forward(output, labels, **kwargs)
