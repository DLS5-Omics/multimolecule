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

from transformers.utils import logging

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PreTrainedConfig

logger = logging.get_logger(__name__)


class RnaMsmConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RnaMsmModel`][multimolecule.models.RnaMsmModel].
    It is used to instantiate a RnaMsm model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RnaMsm
    [yikunpku/RNA-MSM](https://github.com/yikunpku/RNA-MSM) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the RnaMsm model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RnaMsmModel`].
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
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

    Examples:
        >>> from multimolecule import RnaMsmModel, RnaMsmConfig
        >>> # Initializing a RNA-MSM multimolecule/rnamsm style configuration
        >>> configuration = RnaMsmConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/rnamsm style configuration
        >>> model = RnaMsmModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "rnamsm"

    def __init__(
        self,
        vocab_size: int = 26,
        hidden_size: int = 768,
        num_hidden_layers: int = 10,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "absolute",
        is_decoder: bool = False,
        use_cache: bool = True,
        max_tokens_per_msa: int = 2**14,
        layer_type: str = "standard",
        attention_type: str = "standard",
        embed_positions_msa: bool = True,
        attention_bias: bool = True,
        head: HeadConfig | None = None,
        lm_head: MaskedLMHeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.max_tokens_per_msa = max_tokens_per_msa
        self.layer_type = layer_type
        self.attention_type = attention_type
        self.embed_positions_msa = embed_positions_msa
        self.attention_bias = attention_bias
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = MaskedLMHeadConfig(**lm_head) if lm_head is not None else None
