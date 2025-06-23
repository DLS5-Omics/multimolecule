# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
from warnings import warn

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.activations import ACT2FN

from ..criterions import CRITERIONS
from .config import HeadConfig
from .output import HeadOutput
from .transform import HEAD_TRANSFORMS_HF

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


class BasePredictionHead(nn.Module):
    r"""
    Head for all-level of tasks.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    config: HeadConfig
    r"""The configuration object for the head."""

    num_labels: int
    r"""Number of labels for the head."""

    output_name: str | None
    r"""The default output to use for the head."""

    require_attentions: bool = False
    r"""Whether the head requires attentions from the model."""

    bos_token_id: int | None = None
    r"""The ID of the beginning-of-sequence token. Usually is an alias of `cls_token_id`."""

    pad_token_id: int | None = None
    r"""The ID of the padding token."""

    eos_token_id: int | None = None
    r"""The ID of the end-of-sequence token. In rare cases, it is an alias of `sep_token_id`."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__()
        if head_config is None:
            head_config = config.head or HeadConfig(num_labels=config.num_labels)
        if not isinstance(head_config, HeadConfig):
            head_config = HeadConfig(head_config)
        if not head_config.num_labels:
            head_config.num_labels = config.num_labels
        if not head_config.hidden_size:
            head_config.hidden_size = config.hidden_size
        if not head_config.problem_type:
            head_config.problem_type = config.problem_type
        self.config = head_config
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.num_labels = self.config.num_labels  # type: ignore[assignment]
        if getattr(self.config, "output_name", None) is not None:
            self.output_name = self.config.output_name

    def get_attention_mask(self, input_ids: NestedTensor | Tensor) -> Tensor:
        r"""
        Generate attention mask from input IDs or extract from NestedTensor.

        Creates a binary attention mask indicating which tokens should be attended to (1)
        and which should be ignored (0, typically padding tokens). For NestedTensor inputs,
        extracts the pre-computed mask. For regular tensors, compares against pad_token_id.

        Args:
            input_ids: Input token IDs as either a NestedTensor with embedded mask
                or a regular Tensor of shape `(batch_size, seq_len)`.

        Returns:
            Binary attention mask of shape `(batch_size, seq_len)` where 1 indicates
            tokens to attend to and 0 indicates tokens to ignore.

        Raises:
            ValueError: If input_ids is None or if pad_token_id is None when needed
                for regular Tensor inputs.

        Examples:
            >>> import torch
            >>> from multimolecule.models.configuration_utils import PreTrainedConfig
            >>> from multimolecule.modules.heads.generic import BasePredictionHead
            >>> head = BasePredictionHead(PreTrainedConfig(num_labels=2, hidden_size=128))
            >>> input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
            >>> mask = head.get_attention_mask(input_ids)
            >>> mask
            tensor([[1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]], dtype=torch.int32)
        """
        if isinstance(input_ids, NestedTensor):
            return input_ids.mask
        if input_ids is None:
            raise ValueError(
                f"Unable to infer attention mask for {self.__class__.__name__}, because input_ids is None."
            )
        if self.pad_token_id is None:
            raise ValueError(
                f"Unable to infer attention mask for {self.__class__.__name__}, because pad_token_id is None."
            )
        return input_ids.ne(self.pad_token_id).int()

    def remove_special_tokens(
        self, output: Tensor, attention_mask: Tensor, input_ids: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Remove special tokens and clean up model outputs using attention masks.

        Processes model outputs by removing special tokens that were added during tokenization
        and applies attention masking to zero out padding positions. This comprehensive cleanup
        is essential for sequence-level tasks where predictions should only cover the actual
        input sequence, excluding special tokens and padding.

        The method performs:
        - BOS token removal: Strips the first token from all sequences
        - EOS token removal: Strips tokens after the EOS token and the EOS token itself
        - Attention mask adjustment: Updates mask to match the trimmed sequences
        - Output cleanup: Multiplies output by attention mask to zero out padding positions

        Args:
            output: Model output tensor of shape `(batch_size, seq_len, hidden_size)`.
            attention_mask: Attention mask of shape `(batch_size, seq_len)`.
            input_ids: Optional input token IDs of shape `(batch_size, seq_len)`.
                Used for precise EOS token location when available.

        Returns:
            Tuple containing:
                - output: Cleaned output tensor with special tokens removed and padding zeroed
                - attention_mask: Updated attention mask matching trimmed sequences
                - input_ids: Trimmed input IDs (if provided) or unchanged input

        Examples:
            >>> import torch
            >>> from multimolecule.models.configuration_utils import PreTrainedConfig
            >>> from multimolecule.modules.heads.generic import BasePredictionHead
            >>> head = BasePredictionHead(PreTrainedConfig(num_labels=2, hidden_size=4))
            >>> output = torch.randn(1, 6, 4)
            >>> attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
            >>> input_ids = torch.tensor([[1, 10, 20, 2, 0, 0]])
            >>> new_out, new_mask, new_ids = head.remove_special_tokens(output, attention_mask, input_ids)
            >>> output.shape[1], new_out.shape[1]
            (6, 4)
            >>> new_mask
            tensor([[1, 1, 0, 0]])
        """
        if self.bos_token_id is not None:
            output = output[..., 1:, :]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output.device)
                if isinstance(input_ids, Tensor):
                    input_ids.masked_fill_(~eos_mask, self.pad_token_id or 0)
                if isinstance(eos_mask, NestedTensor):
                    eos_mask = eos_mask.tensor
                input_ids = input_ids[..., :-1]
            else:
                last_valid_indices = attention_mask.sum(dim=-1) - 1
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) != last_valid_indices.unsqueeze(1)
            output = (output * eos_mask.unsqueeze(-1))[..., :-1, :]
            attention_mask = (attention_mask * eos_mask)[..., :-1]
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        return output, attention_mask, input_ids

    def remove_special_tokens_2d(
        self, output: Tensor, attention_mask: Tensor, input_ids: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Remove special tokens from 2D outputs like contact maps or pairwise interaction matrices.

        Extends `remove_special_tokens` to handle 2D outputs where both sequence dimensions
        need special token removal. This is crucial for contact prediction and structure
        analysis tasks where the output represents pairwise relationships between residues.

        The method removes:
        - BOS tokens: Strips first row and column from the 2D output
        - EOS tokens: Strips rows/columns after EOS positions and the EOS positions themselves
        - Updates attention mask: Creates 2D mask from 1D sequence mask

        Args:
            output: 2D model output of shape `(batch_size, seq_len, seq_len, channels)`.
            attention_mask: 1D attention mask of shape `(batch_size, seq_len)`.
            input_ids: Optional input token IDs of shape `(batch_size, seq_len)`.

        Returns:
            Tuple containing:
                - output: Trimmed 2D output with special tokens removed from both dimensions
                - attention_mask: 2D attention mask of shape `(batch_size, new_len, new_len)`
                - input_ids: Trimmed input IDs (if provided) or unchanged input

        Examples:
            >>> import torch
            >>> from multimolecule.models.configuration_utils import PreTrainedConfig
            >>> from multimolecule.modules.heads.generic import BasePredictionHead
            >>> head = BasePredictionHead(PreTrainedConfig(num_labels=2, hidden_size=4))
            >>> output = torch.randn(1, 5, 5, 1)
            >>> input_ids = torch.tensor([[1, 10, 20, 2, 0]])
            >>> attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
            >>> new_out, new_mask, new_ids = head.remove_special_tokens_2d(output, attention_mask, input_ids)
            >>> output.shape, new_out.shape
            (torch.Size([1, 5, 5, 1]), torch.Size([1, 3, 3, 1]))
            >>> new_mask.shape
            torch.Size([1, 3, 3])
        """
        if self.bos_token_id is not None:
            output = output[..., 1:, 1:, :]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(output.device)
                if isinstance(input_ids, Tensor):
                    input_ids.masked_fill_(~eos_mask, self.pad_token_id or 0)
                if isinstance(eos_mask, NestedTensor):
                    eos_mask = eos_mask.tensor
                input_ids = input_ids[..., :-1]
            else:
                last_valid_indices = attention_mask.sum(dim=-1) - 1
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) != last_valid_indices.unsqueeze(1)
            attention_mask = (attention_mask * eos_mask)[..., :-1]
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            output = (output * eos_mask.unsqueeze(-1))[..., :-1, :-1, :]
        attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        return output, attention_mask, input_ids

    @staticmethod
    def symmetrize(x: Tensor) -> Tensor:
        r"""
        Make output symmetric by averaging the tensor with its transpose.

        Args:
            x: Input tensor of shape (batch_size, seq_len, seq_len, channels).

        Returns:
            Symmetric tensor with the same shape as input.

        Examples:
            >>> import torch
            >>> from multimolecule.modules.heads.generic import BasePredictionHead
            >>> x = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
            >>> x.squeeze(-1)
            tensor([[[1., 2.],
                     [3., 4.]]])
            >>> symmetric = BasePredictionHead.symmetrize(x)
            >>> symmetric.squeeze(-1)
            tensor([[[1.0000, 2.5000],
                     [2.5000, 4.0000]]])
            >>> torch.allclose(symmetric, symmetric.transpose(1, 2))
            True
        """
        return (x + x.transpose(1, 2)) / 2

    @staticmethod
    def average_product_correct(x: Tensor) -> Tensor:
        r"""Perform Average Product Correction (APC) to remove systematic biases from contact maps.

        APC removes row and column biases that arise from varying residue frequencies and
        structural preferences in molecular contact maps. It subtracts the expected contact
        probability based on marginal frequencies to reveal genuine structural interactions.

        The correction formula: `corrected = original - (row_sums × col_sums) / total_sum`

        This is essential for accurate contact prediction across DNA, RNA, and protein structures.

        Args:
            x: Contact map tensor of shape `(batch_size, seq_len, seq_len, channels)`

        Returns:
            Bias-corrected contact map with the same shape as input

        Note:
            This correction removes spurious correlations caused by sequence composition bias,
            making genuine molecular contacts stand out more clearly from background noise.

        Examples:
            >>> import torch
            >>> from multimolecule.modules.heads.generic import BasePredictionHead
            >>> x = torch.tensor([[[[0.8, 0.6], [0.7, 0.5]], [[0.6, 0.4], [0.5, 0.3]]]])
            >>> x.squeeze(-1)
            tensor([[[[0.8000, 0.6000],
                      [0.7000, 0.5000]],
            <BLANKLINE>
                     [[0.6000, 0.4000],
                      [0.5000, 0.3000]]]])
            >>> corrected = BasePredictionHead.average_product_correct(x)
            >>> corrected.squeeze(-1)
            tensor([[[[-0.0077, -0.0111],
                      [ 0.0077,  0.0111]],
            <BLANKLINE>
                     [[ 0.0077,  0.0111],
                      [-0.0077, -0.0111]]]])
            >>> row_sums = corrected.sum(dim=2).squeeze()
            >>> col_sums = corrected.sum(dim=1).squeeze()
            >>> torch.allclose(row_sums, torch.tensor([[0.0, 0.0], [-0.0, -0.0]]), atol=1e-6)
            True
            >>> torch.allclose(col_sums, torch.tensor([[0.0, 0.0], [-0.0, -0.0]]), atol=1e-6)
            True
        """
        return x - x.sum(1, keepdims=True) * x.sum(2, keepdims=True) / x.sum((1, 2), keepdims=True)


class PredictionHead(BasePredictionHead):
    r"""
    Head for all-level of tasks.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HEAD_TRANSFORMS_HF.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = CRITERIONS.build(self.config)

    def forward(self, embeddings: Tensor, labels: Tensor | None, **kwargs) -> HeadOutput:
        r"""
        Forward pass of the PredictionHead.

        Args:
            embeddings: The embeddings to be passed through the head.
            labels: The labels for the head.
        """
        if kwargs:
            warn(
                f"The following arguments are not applicable to {self.__class__.__name__} "
                f"and will be ignored: {kwargs.keys()}"
            )
        output = self.dropout(embeddings)
        output = self.transform(output)
        output = self.decoder(output)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(output, Tensor):
                    output = labels.nested_like(output, strict=False)
                return HeadOutput(output, self.criterion(output.concat, labels.concat))
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)
