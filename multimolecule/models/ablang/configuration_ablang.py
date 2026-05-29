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

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PreTrainedConfig, validate_attention_dimensions


class AbLangConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an
    [`AbLangModel`][multimolecule.models.AbLangModel]. It is used to instantiate an AbLang v1 model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    configuration similar to the official AbLang v1 heavy/light checkpoints.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the AbLang model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`AbLangModel`].
        hidden_size:
            Dimensionality of the encoder layers and the pooler output.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the feed-forward layer in the Transformer encoder.
        hidden_act:
            Non-linear activation function used by the feed-forward layer and masked language modeling head.
        hidden_dropout:
            Dropout probability applied after embeddings, self-attention, and feed-forward projections.
        attention_dropout:
            Dropout probability applied to attention probabilities.
        max_position_embeddings:
            Size of the learned absolute position embedding table. Position id `0` is reserved for padding.
        initializer_range:
            Standard deviation of the normal initializer for embedding and linear layers.
        layer_norm_eps:
            Epsilon used by layer normalization layers.
        chain:
            Optional antibody chain label for converted checkpoints. AbLang v1 provides separate `heavy` and `light`
            checkpoints trained on different data.
        head:
            The configuration of the downstream prediction head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule.models.ablang import AbLangConfig, AbLangModel
        >>> configuration = AbLangConfig()
        >>> model = AbLangModel(configuration)
        >>> configuration = model.config
    """

    model_type = "ablang"
    position_embedding_type = "absolute"

    def __init__(
        self,
        vocab_size: int = 37,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 160,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-12,
        chain: str | None = None,
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
        validate_attention_dimensions(hidden_size, num_attention_heads)
        hidden_act = hidden_act.lower()
        if max_position_embeddings <= 1:
            raise ValueError("max_position_embeddings must be greater than 1 because position id 0 is padding.")

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
        self.chain = chain
        self.position_embedding_type = "absolute"
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = (
            MaskedLMHeadConfig(**lm_head)
            if lm_head is not None
            else MaskedLMHeadConfig(
                transform="nonlinear",
                transform_act=hidden_act,
                bias=True,
                layer_norm_eps=layer_norm_eps,
            )
        )
