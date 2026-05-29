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


class ProteinBertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`ProteinBertModel`][multimolecule.models.ProteinBertModel]. It is used to instantiate a ProteinBERT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the official ProteinBERT checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the ProteinBERT model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`ProteinBertModel`].
        hidden_size:
            Dimensionality of the local residue representations.
        global_hidden_size:
            Dimensionality of the global protein representation.
        annotation_size:
            Number of Gene Ontology annotation channels used by the pretraining objective.
        num_hidden_layers:
            Number of ProteinBERT local/global encoder blocks.
        num_attention_heads:
            Number of global-attention heads in each encoder block.
        attention_key_size:
            Dimensionality of each global-attention query/key head.
        conv_kernel_size:
            Width of the local convolution kernels.
        wide_conv_dilation_rate:
            Dilation rate of the wide local convolution branch.
        hidden_act:
            Non-linear activation function used by dense and convolutional branches.
        initializer_range:
            Standard deviation used by common prediction heads.
        layer_norm_eps:
            Epsilon used by layer normalization layers.
        head:
            The configuration of the downstream prediction head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import ProteinBertConfig, ProteinBertModel
        >>> configuration = ProteinBertConfig()
        >>> model = ProteinBertModel(configuration)
        >>> configuration = model.config
    """

    model_type = "proteinbert"

    def __init__(
        self,
        vocab_size: int = 37,
        hidden_size: int = 128,
        global_hidden_size: int = 512,
        annotation_size: int = 8943,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
        attention_key_size: int = 64,
        conv_kernel_size: int = 9,
        wide_conv_dilation_rate: int = 5,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-3,
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
        if global_hidden_size % num_attention_heads != 0:
            raise ValueError(
                "global_hidden_size must be divisible by num_attention_heads; got "
                f"{global_hidden_size} and {num_attention_heads}."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.global_hidden_size = global_hidden_size
        self.annotation_size = annotation_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_key_size = attention_key_size
        self.conv_kernel_size = conv_kernel_size
        self.wide_conv_dilation_rate = wide_conv_dilation_rate
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = (
            MaskedLMHeadConfig(**lm_head)
            if lm_head is not None
            else MaskedLMHeadConfig(transform=None, transform_act=None, bias=True)
        )
