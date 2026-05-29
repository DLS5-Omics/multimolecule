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

from ..configuration_utils import (
    HeadConfig,
    MaskedLMHeadConfig,
    PreTrainedConfig,
    validate_attention_dimensions,
)


class AbLang2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an
    [`AbLang2Model`][multimolecule.models.AbLang2Model]. It is used to instantiate an AbLang2 model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to the official AbLang2 paired-antibody checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the AbLang2 model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`AbLang2Model`].
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the feed-forward hidden layer after the activation. For SwiGLU, the first feed-forward
            projection has twice this size.
        hidden_act:
            The non-linear activation function used in the feed-forward layers. AbLang2 uses `"swiglu"`.
        hidden_dropout:
            The dropout probability applied to residual and feed-forward outputs.
        attention_dropout:
            The dropout ratio applied to attention probabilities.
        initializer_range:
            Standard deviation used by default initialization for embeddings and linear layers.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        rotary_base:
            Base used for rotary position embeddings.
        attention_bias:
            Whether to use bias terms in the attention projections.
        feedforward_bias:
            Whether to use bias terms in the feed-forward projections.
        head:
            The configuration of the downstream prediction head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import AbLang2Config, AbLang2Model
        >>> # Initializing an AbLang2 multimolecule/ablang2 style configuration
        >>> configuration = AbLang2Config()
        >>> # Initializing a model (with random weights) from the multimolecule/ablang2 style configuration
        >>> model = AbLang2Model(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "ablang2"

    def __init__(
        self,
        vocab_size: int = 37,
        hidden_size: int = 480,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 20,
        intermediate_size: int = 1920,
        hidden_act: str = "swiglu",
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-12,
        rotary_base: float = 10000.0,
        attention_bias: bool = True,
        feedforward_bias: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        mask_token_id: int = 4,
        null_token_id: int = 5,
        sep_token_id: int = 32,
        head: HeadConfig | None = None,
        lm_head: MaskedLMHeadConfig | None = None,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            mask_token_id=mask_token_id,
            null_token_id=null_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )
        validate_attention_dimensions(hidden_size, num_attention_heads)
        hidden_act = hidden_act.lower()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_base = rotary_base
        self.attention_bias = attention_bias
        self.feedforward_bias = feedforward_bias
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = (
            MaskedLMHeadConfig(**lm_head)
            if lm_head is not None
            else MaskedLMHeadConfig(
                hidden_size=hidden_size,
                transform=None,
                transform_act=hidden_act,
                bias=True,
                layer_norm_eps=layer_norm_eps,
            )
        )
