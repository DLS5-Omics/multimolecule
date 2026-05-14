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


class AMPLIFYConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AMPLIFYModel`][multimolecule.models.AMPLIFYModel].
    It is used to instantiate an AMPLIFY model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the AMPLIFY
    [chandar-lab/AMPLIFY_120M](https://huggingface.co/chandar-lab/AMPLIFY_120M) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the AMPLIFY model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`AMPLIFYModel`].
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act:
            The non-linear activation function used in the feed-forward layer. AMPLIFY uses `"swiglu"` by default.
        hidden_dropout:
            The dropout probability applied to residual connections and feed-forward outputs.
        attention_dropout:
            The dropout ratio applied to attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with. Used to precompute rotary
            frequencies.
        initializer_range:
            The standard deviation (or half-range, for uniform init) of the initializer for initializing all
            weight matrices.
        layer_norm_eps:
            The epsilon used by the RMSNorm/LayerNorm layers.
        rms_norm:
            Whether to use RMSNorm instead of LayerNorm.
        layer_norm_after_embedding:
            Whether to apply a normalization layer right after the token embedding.
        layer_norm_before_last_layer:
            Whether to apply a normalization layer before the output projection.
        attention_bias:
            Whether to use bias terms in the attention projections (`q`, `k`, `v`, `out_proj`).
        feedforward_bias:
            Whether to use bias terms in the feed-forward projections.
        head:
            The configuration of the head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import AMPLIFYConfig, AMPLIFYModel
        >>> # Initializing an AMPLIFY 120M style configuration
        >>> configuration = AMPLIFYConfig()
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = AMPLIFYModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "amplify"

    def __init__(
        self,
        vocab_size: int = 27,
        hidden_size: int = 640,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 10,
        intermediate_size: int = 2560,
        hidden_act: str = "swiglu",
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-5,
        rms_norm: bool = True,
        layer_norm_after_embedding: bool = False,
        layer_norm_before_last_layer: bool = True,
        attention_bias: bool = False,
        feedforward_bias: bool = False,
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
        # AMPLIFY stores ``encoder.weight`` and ``decoder.weight`` as separate
        # parameters; weight tying must remain disabled to preserve checkpoint
        # behaviour.
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
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})."
            )
        if (hidden_size // num_attention_heads) % 2 != 0:
            raise ValueError(
                "Rotary embeddings require an even head dimension; got "
                f"hidden_size={hidden_size}, num_attention_heads={num_attention_heads}."
            )
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
        self.rms_norm = rms_norm
        self.layer_norm_after_embedding = layer_norm_after_embedding
        self.layer_norm_before_last_layer = layer_norm_before_last_layer
        self.attention_bias = attention_bias
        self.feedforward_bias = feedforward_bias
        self.head = HeadConfig(**head) if head is not None else None
        # AMPLIFY's LM head is a single ``Linear`` with bias; no projection/norm
        # transform sits between the encoder output and the decoder.
        self.lm_head = MaskedLMHeadConfig(**lm_head) if lm_head is not None else MaskedLMHeadConfig(
            transform=None, transform_act=None, bias=True
        )
