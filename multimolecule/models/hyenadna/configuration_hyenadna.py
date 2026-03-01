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

from ..configuration_utils import HeadConfig, PreTrainedConfig


class HyenaDnaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`HyenaDnaModel`][multimolecule.models.HyenaDnaModel]. It is used to instantiate a HyenaDNA model according
    to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the HyenaDNA
    [LongSafari/hyenadna-medium-160k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf)
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the HyenaDNA model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`HyenaDnaModel`].
        hidden_size:
            Dimensionality of the model layers.
        num_hidden_layers:
            Number of hidden layers (Hyena blocks) in the model.
        intermediate_size:
            Dimensionality of the feed-forward layer. If `None`, defaults to `4 * hidden_size`.
        embedding_dropout:
            The dropout probability for the embedding layer.
        hidden_dropout:
            The dropout probability within the Hyena operator.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with.
        initializer_range:
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        hyena_order:
            Order of the Hyena recurrence. Controls the number of element-wise gating steps.
        filter_order:
            Width of the implicit filter MLP (number of hidden units).
        short_filter_order:
            Kernel size of the short depthwise convolution applied before the Hyena recurrence.
        filter_emb_dim:
            Dimensionality of the positional embedding fed to the implicit filter MLP.
            Must be odd and >= 3. Computed as `(1 time) + (2 * num_frequency_bands)`.
        num_inner_mlps:
            Number of inner linear layers inside the implicit filter MLP.
        activation_freq:
            Frequency multiplier for the Sin activation function in the implicit filter.
        filter_dropout:
            The dropout probability for the implicit filter.
        use_bias:
            Whether to use bias in the implicit filter.
        train_freq:
            Whether the Sin activation frequencies are learnable parameters.
        pad_vocab_size_multiple:
            Pad the vocabulary size to be a multiple of this value (for GPU performance).
        head:
            The configuration of the head.

    Examples:
        >>> from multimolecule import HyenaDnaConfig, HyenaDnaModel
        >>> # Initializing a HyenaDNA multimolecule/hyenadna style configuration
        >>> configuration = HyenaDnaConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/hyenadna style configuration
        >>> model = HyenaDnaModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "hyenadna"

    def __init__(
        self,
        vocab_size: int = 12,
        hidden_size: int = 256,
        num_hidden_layers: int = 8,
        intermediate_size: int | None = None,
        embedding_dropout: float = 0.1,
        hidden_dropout: float = 0.0,
        max_position_embeddings: int = 160002,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        hyena_order: int = 2,
        filter_order: int = 64,
        short_filter_order: int = 3,
        filter_emb_dim: int = 5,
        num_inner_mlps: int = 2,
        activation_freq: int = 10,
        filter_dropout: float = 0.0,
        use_bias: bool = True,
        train_freq: bool = True,
        pad_vocab_size_multiple: int = 8,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.embedding_dropout = embedding_dropout
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hyena_order = hyena_order
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.filter_emb_dim = filter_emb_dim
        self.num_inner_mlps = num_inner_mlps
        self.activation_freq = activation_freq
        self.filter_dropout = filter_dropout
        self.use_bias = use_bias
        self.train_freq = train_freq
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.head = HeadConfig(**head) if head is not None else None
