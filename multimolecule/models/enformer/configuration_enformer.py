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


class EnformerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`EnformerModel`][multimolecule.models.EnformerModel]. It is used to instantiate an Enformer model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the Enformer
    [deepmind/enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Enformer is the successor of Basenji. It replaces Basenji's dilated convolution tower with a
    convolution stem followed by a Transformer trunk so it can model long-range genomic
    interactions. A long DNA window of `sequence_length` base pairs is downsampled by the
    convolution stem (`2 ** num_downsamples`, i.e. 128 bp per bin by default), processed by the
    Transformer trunk, cropped to `target_length` bins, and projected to genomic coverage tracks.
    The output has shape `(batch_size, target_length, num_labels)`.

    Args:
        vocab_size:
            Vocabulary size of the Enformer model. Defines the number of input feature channels
            derived from the MultiMolecule DNA token order.
            Defaults to 5 (`A`, `C`, `G`, `T`, `N`).
        sequence_length:
            The length, in base pairs, of the input DNA window.
            Defaults to 196608 (~197 kb), matching the released Enformer checkpoint.
        hidden_size:
            Dimensionality of the Transformer trunk. The convolution stem's first conv produces
            `hidden_size // 2` channels and the conv tower grows back to `hidden_size`.
        num_hidden_layers:
            Number of Transformer blocks in the trunk.
        num_attention_heads:
            Number of attention heads in each Transformer block.
        attention_head_size:
            Dimensionality of the query/key projection per head.
        num_downsamples:
            Total number of 2x downsampling steps applied by the convolution stem. The binning
            factor is `2 ** num_downsamples` (128 bp per bin by default).
        dim_divisible_by:
            The conv-tower channel sizes are rounded to a multiple of this value.
        stem_kernel_size:
            Kernel size of the first (stem) convolution.
        conv_tower_kernel_size:
            Kernel size of the main convolution in every conv-tower stage.
        target_length:
            Number of output bins kept after center-cropping the trunk output.
        head_hidden_size:
            Dimensionality of the pointwise output head before the final track projection.
            Defaults to `2 * hidden_size`.
        hidden_act:
            The non-linear activation function used by the convolution blocks and the pointwise
            head. Enformer uses the sigmoid GELU approximation `x * sigmoid(1.702 * x)`, which is
            `quick_gelu` in Transformers.
        output_act:
            Activation applied to the per-track predictions. Enformer applies `softplus` so the
            predicted coverage is non-negative.
        hidden_dropout:
            The dropout probability applied in the Transformer trunk.
        attention_dropout:
            The dropout probability applied to the attention matrix.
        position_dropout:
            The dropout probability applied to the relative positional features.
        use_precomputed_gamma_basis:
            Whether to use the fixed precomputed gamma relative-position basis distributed with
            the released checkpoint. The official converted checkpoint stores this basis table.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        species:
            Output head to expose downstream. Enformer is trained with two species heads; the
            selected head determines `num_labels`. Use `human` (5313 tracks) or `mouse`
            (1643 tracks).
        num_labels:
            Number of genomic coverage tracks predicted per bin. Defaults to the track count of
            the selected `species` head.
        head:
            Head configuration for the binned track prediction head.
        output_contexts:
            Whether to output the context vectors for each trunk block.

    Examples:
        >>> from multimolecule import EnformerConfig, EnformerModel
        >>> # Initializing an Enformer multimolecule/enformer style configuration
        >>> configuration = EnformerConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/enformer style configuration
        >>> model = EnformerModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "enformer"

    species_num_tracks = {"human": 5313, "mouse": 1643}

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 196608,
        hidden_size: int = 1536,
        num_hidden_layers: int = 11,
        num_attention_heads: int = 8,
        attention_head_size: int = 64,
        num_downsamples: int = 7,
        dim_divisible_by: int = 128,
        stem_kernel_size: int = 15,
        conv_tower_kernel_size: int = 5,
        target_length: int = 896,
        head_hidden_size: int | None = None,
        hidden_act: str = "quick_gelu",
        output_act: str = "softplus",
        hidden_dropout: float = 0.4,
        attention_dropout: float = 0.05,
        position_dropout: float = 0.01,
        use_precomputed_gamma_basis: bool = False,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        species: str = "human",
        num_labels: int | None = None,
        head: HeadConfig | None = None,
        output_contexts: bool = False,
        **kwargs,
    ):
        # Enformer is a feature-channel DNA model: it consumes a raw one-hot DNA window with no
        # special tokens, and its output is binned coverage tracks. There is no BOS/EOS/MASK token
        # on either the input or the binned positional axis, so the shared TokenPredictionHead must
        # not trim "special" bins.
        kwargs.setdefault("bos_token_id", None)
        kwargs.setdefault("eos_token_id", None)
        kwargs.setdefault("mask_token_id", None)
        kwargs.setdefault("null_token_id", None)
        if species not in self.species_num_tracks:
            raise ValueError(f"species must be one of {sorted(self.species_num_tracks)}, got {species!r}")
        if num_labels is None:
            num_labels = self.species_num_tracks[species]
        super().__init__(num_labels=num_labels, **kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.stem_kernel_size = stem_kernel_size
        self.conv_tower_kernel_size = conv_tower_kernel_size
        self.target_length = target_length
        self.head_hidden_size = head_hidden_size if head_hidden_size is not None else 2 * hidden_size
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.position_dropout = position_dropout
        self.use_precomputed_gamma_basis = use_precomputed_gamma_basis
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.species = species
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
        self.output_contexts = output_contexts

        if self.num_downsamples < 2:
            raise ValueError(f"num_downsamples must be >= 2, got {self.num_downsamples}")
        if self.hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even, got {self.hidden_size}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})"
            )
        # The relative positional encoding stacks 3 basis families x 2 (signed) = 6 components, so
        # the per-head feature size must be divisible by 6.
        if (self.hidden_size // self.num_attention_heads) % 6 != 0:
            raise ValueError(
                f"hidden_size // num_attention_heads "
                f"({self.hidden_size // self.num_attention_heads}) must be divisible by 6 so the "
                f"relative positional features are well defined."
            )
        if self.pool_factor <= 0:
            raise ValueError(f"pool_factor must be positive, got {self.pool_factor}")
        if self.sequence_length % self.pool_factor != 0:
            raise ValueError(
                f"sequence_length ({self.sequence_length}) must be divisible by the total pooling "
                f"factor ({self.pool_factor}) so the binned output is well defined."
            )
        if self.num_bins < self.target_length:
            raise ValueError(
                f"target_length ({self.target_length}) must not exceed the number of binned "
                f"positions ({self.num_bins})."
            )

    @property
    def pool_factor(self) -> int:
        r"""Total downsampling factor applied by the stem, i.e. base pairs per output bin."""
        return 2**self.num_downsamples

    @property
    def num_bins(self) -> int:
        r"""Number of binned positions produced by the trunk before center-cropping."""
        return self.sequence_length // self.pool_factor
