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

from transformers.utils import logging

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PreTrainedConfig

logger = logging.get_logger(__name__)


class RibonanzaNetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`RibonanzaNetModel`][multimolecule.models.RibonanzaNetModel].
    It is used to instantiate a RibonanzaNet model according to the specified arguments, defining the model
    architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RibonanzaNet
    [Shujun-He/RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the RibonanzaNet model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`RibonanzaNetModel`].
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout:
            The dropout ratio for the attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range:
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        position_embedding_type:
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder:
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache:
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before:
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout:
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
        head:
            The configuration of the head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetModel
        >>> # Initializing a RibonanzaNet multimolecule/ribonanzanet style configuration
        >>> configuration = RibonanzaNetConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/ribonanzanet style configuration
        >>> model = RibonanzaNetModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "ribonanzanet"

    def __init__(
        self,
        vocab_size: int = 26,
        hidden_size: int = 256,
        num_hidden_layers: int = 9,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        pairwise_size: int = 64,
        pairwise_attention_size: int = 32,
        pairwise_intermediate_size: int = 256,
        pairwise_num_attention_heads: int = 4,
        kernel_size: int = 5,
        use_triangular_attention: bool = False,
        hidden_act: str = "gelu",
        pairwise_hidden_act: str = "relu",
        hidden_dropout: float = 0.05,
        attention_dropout: float = 0.05,
        output_pairwise_states: bool = False,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        head: HeadConfig | None = None,
        lm_head: MaskedLMHeadConfig | None = None,
        fix_attention_mask: bool = False,
        fix_attention_norm: bool = False,
        fix_pairwise_dropout: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.type_vocab_size = 2
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.pairwise_size = pairwise_size
        self.pairwise_attention_size = pairwise_attention_size
        self.pairwise_intermediate_size = pairwise_intermediate_size
        self.pairwise_num_attention_heads = pairwise_num_attention_heads
        self.kernel_size = kernel_size
        self.use_triangular_attention = use_triangular_attention
        self.hidden_act = hidden_act
        self.pairwise_hidden_act = pairwise_hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.output_pairwise_states = output_pairwise_states
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = MaskedLMHeadConfig(**lm_head) if lm_head is not None else None
        self.fix_attention_mask = fix_attention_mask
        self.fix_attention_norm = fix_attention_norm
        self.fix_pairwise_dropout = fix_pairwise_dropout
