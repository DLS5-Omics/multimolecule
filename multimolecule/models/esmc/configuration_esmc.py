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

import math

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PreTrainedConfig, validate_attention_dimensions


class EsmCConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an
    [`EsmCModel`][multimolecule.models.EsmCModel]. It is used to instantiate an ESMC model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to Biohub's [biohub/ESMC-300M](https://huggingface.co/biohub/ESMC-300M) architecture after conversion
    to the MultiMolecule protein vocabulary.

    Args:
        vocab_size:
            Vocabulary size of the ESMC model. MultiMolecule checkpoints use the standard
            [`ProteinTokenizer`][multimolecule.ProteinTokenizer] vocabulary.
        hidden_size:
            Dimensionality of the encoder layers.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer.
        intermediate_size:
            Dimensionality of the SwiGLU hidden projection after splitting the packed upstream projection. If `None`,
            it is computed from `hidden_size` using ESMC's `8 / 3` expansion rounded up to a multiple of 256.
        hidden_act:
            The non-linear activation function used in the feed-forward layer. ESMC uses `"swiglu"`.
        hidden_dropout:
            Dropout probability applied to residual branches.
        attention_dropout:
            Dropout probability applied to attention probabilities.
        max_position_embeddings:
            Maximum sequence length used for documentation and tokenizer compatibility. ESMC uses rotary embeddings,
            so this does not allocate learned position embeddings.
        initializer_range:
            Standard deviation of the initializer for weight matrices.
        layer_norm_eps:
            Epsilon used by layer normalization.
        attention_bias:
            Whether to use bias terms in attention projection layers.
        feedforward_bias:
            Whether to use bias terms in feed-forward projection layers.
        attention_layer_norm_bias:
            Whether to use a bias in the pre-attention layer normalization.
        qk_layer_norm:
            Whether to apply layer normalization to query and key projections before rotary embeddings.
        qk_layer_norm_bias:
            Whether to use a bias in query/key layer normalization.
        final_layer_norm_bias:
            Whether to use a bias in the final encoder layer normalization.
        residue_scaling_factor:
            Residual branch divisor used by ESMC. If `None`, defaults to `sqrt(num_hidden_layers / 36)`.
        head:
            The configuration of the downstream prediction heads.
        lm_head:
            The configuration of the masked language model head.
    """

    model_type = "esmc"

    def __init__(
        self,
        vocab_size: int = 37,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "swiglu",
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-5,
        attention_bias: bool = False,
        feedforward_bias: bool = False,
        attention_layer_norm_bias: bool = True,
        qk_layer_norm: bool = True,
        qk_layer_norm_bias: bool = False,
        final_layer_norm_bias: bool = False,
        residue_scaling_factor: float | None = None,
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
        hidden_size = int(kwargs.pop("d_model", 960) if hidden_size is None else hidden_size)
        num_hidden_layers = int(kwargs.pop("n_layers", 30) if num_hidden_layers is None else num_hidden_layers)
        num_attention_heads = int(kwargs.pop("n_heads", 15) if num_attention_heads is None else num_attention_heads)
        intermediate_size = (
            _swiglu_intermediate_size(hidden_size) if intermediate_size is None else int(intermediate_size)
        )
        residue_scaling_factor = (
            math.sqrt(num_hidden_layers / 36) if residue_scaling_factor is None else residue_scaling_factor
        )
        # ESMC stores the token embedding and MLM decoder as independent parameters.
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
        if (hidden_size // num_attention_heads) % 2 != 0:
            raise ValueError(
                "Rotary embeddings require an even head dimension; got "
                f"hidden_size={hidden_size}, num_attention_heads={num_attention_heads}."
            )
        hidden_act = hidden_act.lower()
        if hidden_act != "swiglu":
            raise ValueError(f"ESMC only supports hidden_act='swiglu'; got {hidden_act!r}.")
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
        self.attention_bias = attention_bias
        self.feedforward_bias = feedforward_bias
        self.attention_layer_norm_bias = attention_layer_norm_bias
        self.qk_layer_norm = qk_layer_norm
        self.qk_layer_norm_bias = qk_layer_norm_bias
        self.final_layer_norm_bias = final_layer_norm_bias
        self.residue_scaling_factor = residue_scaling_factor
        self.head = HeadConfig(**head) if head is not None else None
        self.lm_head = (
            MaskedLMHeadConfig(**lm_head)
            if lm_head is not None
            else MaskedLMHeadConfig(
                transform="nonlinear",
                transform_act="gelu",
                bias=True,
                layer_norm_eps=layer_norm_eps,
            )
        )


def _swiglu_intermediate_size(hidden_size: int, multiple_of: int = 256, expansion_ratio: float = 8 / 3) -> int:
    """Round ESMC's SwiGLU hidden size up to the next ``multiple_of``."""
    return multiple_of * math.ceil(expansion_ratio * hidden_size / multiple_of)
