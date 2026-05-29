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

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PreTrainedConfig


class CarpConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CarpModel`][multimolecule.models.CarpModel].
    It is used to instantiate a CARP model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to the official `carp_600k`
    checkpoint from Microsoft Research.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the CARP model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`CarpModel`].
        embedding_size:
            Dimensionality of the token embeddings before the ByteNet projection.
        hidden_size:
            Dimensionality of the ByteNet residual stream.
        intermediate_size:
            Dimensionality used inside each dilated convolution block. If `None`, CARP uses `hidden_size // 2` when
            `slim=True`, otherwise `hidden_size`.
        num_hidden_layers:
            Number of ByteNet residual convolution blocks.
        kernel_size:
            Width of the dilated one-dimensional convolution kernels.
        max_dilation:
            Largest dilation factor in the cyclic ByteNet dilation schedule.
        hidden_act:
            Non-linear activation function used in each residual convolution block. CARP checkpoints use `"gelu"`.
        hidden_dropout:
            Dropout probability applied after each residual convolution block.
        initializer_range:
            Standard deviation used for newly initialized vocabulary rows during conversion.
        layer_norm_eps:
            Epsilon used by layer normalization layers.
        slim:
            Whether the checkpoint uses the half-width hidden channel inside residual convolution blocks.
        head:
            The configuration of the downstream prediction head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import CarpConfig, CarpModel
        >>> # Initializing a CARP multimolecule/carp style configuration
        >>> configuration = CarpConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/carp style configuration
        >>> model = CarpModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "carp"

    def __init__(
        self,
        vocab_size: int = 37,
        embedding_size: int = 8,
        hidden_size: int = 128,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 16,
        kernel_size: int = 5,
        max_dilation: int = 128,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-5,
        slim: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        mask_token_id: int = 4,
        null_token_id: int = 5,
        head: HeadConfig | None = None,
        lm_head: MaskedLMHeadConfig | None = None,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            mask_token_id=mask_token_id,
            null_token_id=null_token_id,
            **kwargs,
        )

        hidden_act = hidden_act.lower()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer; got {kernel_size}.")
        if max_dilation < 1:
            raise ValueError(f"max_dilation must be positive; got {max_dilation}.")
        if intermediate_size is None:
            intermediate_size = hidden_size // 2 if slim else hidden_size
        for name, value in {
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}.")

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.kernel_size = kernel_size
        self.max_dilation = max_dilation
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.slim = slim
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = (
            MaskedLMHeadConfig(**lm_head)
            if lm_head is not None
            else MaskedLMHeadConfig(transform=None, transform_act=None, bias=True)
        )
