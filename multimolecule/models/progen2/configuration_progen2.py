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

from ..configuration_utils import PreTrainedConfig


class ProGen2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProGen2Model`][multimolecule.models.ProGen2Model].
    It is used to instantiate a ProGen2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ProGen2
    [salesforce/progen2](https://github.com/salesforce/progen) architecture, which follows the GPT-J style transformer.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the ProGen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ProGen2Model`].
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
        embedding_dropout:
            The dropout probability for the embedding layer.
        hidden_dropout:
            The dropout probability for residual connections and fully connected layers in the decoder.
        attention_dropout:
            The dropout ratio for the attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range:
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        rotary_dim:
            Dimensionality of rotary position embeddings. If `None`, rotary embeddings are applied across the full
            head dimension.
        scale_attn_weights:
            Whether to scale attention weights by sqrt(head_dim).
        gradient_checkpointing:
            Whether to use gradient checkpointing.
        use_cache:
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        is_decoder:
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.

    Examples:
        >>> from multimolecule import ProGen2Config, ProGen2Model
        >>> # Initializing a ProGen2 multimolecule/progen2 style configuration
        >>> configuration = ProGen2Config()
        >>> # Initializing a model (with random weights) from the multimolecule/progen2 style configuration
        >>> model = ProGen2Model(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "progen2"

    def __init__(
        self,
        vocab_size: int = 35,
        hidden_size: int = 1536,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        intermediate_size: int | None = None,
        hidden_act: str = "gelu_new",
        embedding_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        rotary_dim: int | None = 48,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        gradient_checkpointing: bool = False,
        is_decoder: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("null_token_id", None)
        super().__init__(**kwargs)
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_dropout = embedding_dropout
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_dim = rotary_dim
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.is_decoder = is_decoder
