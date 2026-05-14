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


class GenerannoConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`GenerannoModel`][multimolecule.models.GenerannoModel]. It is used to instantiate a GENERanno model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the GENERanno
    [GenerTeam/GENERanno-eukaryote-0.5b-base](https://huggingface.co/GenerTeam/GENERanno-eukaryote-0.5b-base)
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the GENERanno model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`GenerannoModel`].
        hidden_size:
            Dimensionality of the encoder layers.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads:
            Number of key/value heads used for grouped-query attention. When `num_key_value_heads == num_attention_heads`
            the model uses multi-head attention; when `num_key_value_heads == 1` it uses multi-query attention.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) SwiGLU layer.
        hidden_act:
            The non-linear activation function used for the SwiGLU gate. `silu` (a.k.a. swish) is the default.
        attention_dropout:
            The dropout ratio for the attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with.
        initializer_range:
            The standard deviation of the truncated normal initializer used to initialize all weight matrices.
        rms_norm_eps:
            The epsilon used by the RMS normalization layers.
        attention_bias:
            Whether to use a bias in the Q, K, V, and output projection layers during self-attention.
        mlp_bias:
            Whether to use a bias in the up, down, and gate projection layers in the SwiGLU MLP.
        rope_theta:
            The base period of the rotary position embeddings.
        head:
            The configuration of the prediction head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import GenerannoConfig, GenerannoModel
        >>> configuration = GenerannoConfig()
        >>> model = GenerannoModel(configuration)
        >>> configuration = model.config
    """

    model_type = "generanno"

    def __init__(
        self,
        vocab_size: int = 26,
        hidden_size: int = 1280,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = 4,
        intermediate_size: int = 3520,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        rope_theta: float = 500000.0,
        head: HeadConfig | None = None,
        lm_head: MaskedLMHeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads}) for grouped-query attention"
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rope_theta = rope_theta
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = MaskedLMHeadConfig(**lm_head) if lm_head is not None else None
