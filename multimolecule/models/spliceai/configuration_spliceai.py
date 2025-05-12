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

from chanfig import FlatDict

from ..configuration_utils import PreTrainedConfig


class SpliceAiStageConfig(FlatDict):
    num_blocks: int = 4
    kernel_size: int = 11
    dilation: int = 1


class SpliceAiConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`SpliceAiModel`][multimolecule.models.SpliceAiModel]. It is used to instantiate a SpliceAI model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the SpliceAI [Illumina/SpliceAI](https://github.com/Illumina/SpliceAI)
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the SpliceAI model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SpliceAiModel`].
            Defaults to 131 if `codon=True` else 26.
        codon:
            Whether to use codon tokenization.
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
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`,
            `"rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
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
        >>> from multimolecule import SpliceAiConfig, SpliceAiModel
        >>> # Initializing a SpliceAI multimolecule/spliceai style configuration
        >>> configuration = SpliceAiConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/spliceai style configuration
        >>> model = SpliceAiModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "spliceai"

    def __init__(
        self,
        vocab_size: int = 4,
        context: int = 10000,
        hidden_size: int = 32,
        stages: list[SpliceAiStageConfig] | None = None,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.1,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        num_labels: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context = context
        self.padding = context // 2
        if stages is None:
            stages = [
                SpliceAiStageConfig(num_blocks=4, kernel_size=11),
                SpliceAiStageConfig(num_blocks=4, kernel_size=11, dilation=4),
                SpliceAiStageConfig(num_blocks=4, kernel_size=21, dilation=10),
                SpliceAiStageConfig(num_blocks=4, kernel_size=41, dilation=25),
            ]
        self.stages = stages
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.num_labels = num_labels
