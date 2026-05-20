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


class BorzoiConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`BorzoiModel`][multimolecule.models.BorzoiModel]. It is used to instantiate a Borzoi model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    configuration that reproduces the upstream Borzoi human architecture
    ([calico/borzoi](https://github.com/calico/borzoi), `examples/params_pred.json`).

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Borzoi is the successor of Enformer. It extends the Enformer recipe (convolution stem + Transformer trunk +
    binned multi-track output) to a 524,288 bp input window and 32 bp output bins, and adds a U-Net style
    upsampling tail so the binned positional axis matches a higher-resolution coverage prediction. A long DNA
    window of `sequence_length` base pairs is downsampled by a convolution stem and a width-growing residual
    convolution tower, projected to `hidden_size` channels by a U-Net bottleneck, processed by the Transformer
    trunk with Transformer-XL style relative positional encoding, then upsampled by two skip-connected U-Net
    stages with depthwise-separable convolutions, center-cropped to `target_length` bins, and projected to
    per-species coverage tracks with a softplus activation. The output has shape
    `(batch_size, target_length, num_labels)`.

    Args:
        vocab_size:
            Vocabulary size of the Borzoi model. Defines the number of input feature channels derived from the
            MultiMolecule DNA token order.
            Defaults to 5 (`A`, `C`, `G`, `T`, `N`).
        sequence_length:
            The length, in base pairs, of the input DNA window.
            Defaults to 524288 (= 512 kb).
        hidden_size:
            Dimensionality of the Transformer trunk and the U-Net upsampling tail.
        num_hidden_layers:
            Number of Transformer blocks in the trunk.
        num_attention_heads:
            Number of attention heads in each Transformer block.
        attention_head_size:
            Dimensionality of the query/key projection per head.
        attention_value_size:
            Dimensionality of the value projection per head. Borzoi uses a larger value dim than key dim.
        num_rel_pos_features:
            Number of relative positional features used by the Transformer-XL style attention.
        stem_channels:
            Number of channels produced by the first (stem) convolution.
        stem_kernel_size:
            Kernel size of the first (stem) convolution.
        conv_tower_channels:
            Explicit per-stage output channel schedule of the reducing convolution tower. Borzoi grows the
            width as ``608, 736, 896, 1056, 1280``; the tower length is ``len(conv_tower_channels)``.
        conv_tower_kernel_size:
            Kernel size used by every convolution in the reducing tower.
        unet_kernel_size:
            Kernel size of the depthwise-separable convolutions in the U-Net upsampling tail.
        head_hidden_size:
            Channel count of the final pointwise convolution block feeding the per-species track head.
        hidden_act:
            The non-linear activation used throughout the convolution blocks. Borzoi uses the tanh-approximation
            GELU (`gelu_new`).
        output_act:
            Activation applied to the per-track predictions. Borzoi applies `softplus` so the predicted coverage
            is non-negative.
        hidden_dropout:
            Dropout probability of the final pointwise convolution block.
        intermediate_dropout:
            Dropout probability applied inside the Transformer feed-forward sublayer.
        attention_dropout:
            Dropout probability applied to the attention matrix.
        position_dropout:
            Dropout probability applied to the relative positional features.
        batch_norm_eps:
            Epsilon used by the batch normalization layers.
        batch_norm_momentum:
            Momentum used by the batch normalization layers (PyTorch convention; upstream Keras momentum 0.9
            corresponds to PyTorch momentum 0.1).
        species:
            Output head to expose downstream. Borzoi is trained with two species heads; the selected head
            determines `num_labels`. Use `human` (7611 tracks) or `mouse` (2608 tracks).
        target_length:
            Number of output bins kept after center-cropping the U-Net output. Defaults to 6144 (the
            `bins_to_return` setting of the upstream Borzoi inference path).
        num_labels:
            Number of genomic coverage tracks predicted per bin. Defaults to the track count of the selected
            `species` head.
        head:
            Head configuration for the binned track prediction head.
        output_contexts:
            Whether to output the context vectors for each trunk block.

    Examples:
        >>> from multimolecule import BorzoiConfig, BorzoiModel
        >>> # Initializing a Borzoi multimolecule/borzoi style configuration
        >>> configuration = BorzoiConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/borzoi style configuration
        >>> model = BorzoiModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "borzoi"

    species_num_tracks = {"human": 7611, "mouse": 2608}

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 524288,
        hidden_size: int = 1536,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        attention_head_size: int = 64,
        attention_value_size: int = 192,
        num_rel_pos_features: int = 32,
        stem_channels: int = 512,
        stem_kernel_size: int = 15,
        conv_tower_channels: list[int] | None = None,
        conv_tower_kernel_size: int = 5,
        unet_kernel_size: int = 3,
        head_hidden_size: int = 1920,
        hidden_act: str = "gelu_new",
        output_act: str = "softplus",
        hidden_dropout: float = 0.1,
        intermediate_dropout: float = 0.2,
        attention_dropout: float = 0.05,
        position_dropout: float = 0.01,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.1,
        species: str = "human",
        target_length: int = 6144,
        num_labels: int | None = None,
        head: HeadConfig | None = None,
        output_contexts: bool = False,
        **kwargs,
    ):
        # Borzoi is a feature-channel DNA model: it consumes a raw one-hot DNA window with no special tokens,
        # and its output is binned coverage tracks. There is no BOS/EOS/MASK token on either the input or the
        # binned positional axis, so the shared TokenPredictionHead must not trim "special" bins.
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
        self.attention_value_size = attention_value_size
        self.num_rel_pos_features = num_rel_pos_features
        self.stem_channels = stem_channels
        self.stem_kernel_size = stem_kernel_size
        if conv_tower_channels is None:
            conv_tower_channels = [608, 736, 896, 1056, 1280]
        self.conv_tower_channels = list(conv_tower_channels)
        self.conv_tower_kernel_size = conv_tower_kernel_size
        self.unet_kernel_size = unet_kernel_size
        self.head_hidden_size = head_hidden_size
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.hidden_dropout = hidden_dropout
        self.intermediate_dropout = intermediate_dropout
        self.attention_dropout = attention_dropout
        self.position_dropout = position_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.species = species
        self.target_length = target_length
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
        self.output_contexts = output_contexts

        if self.stem_channels < 1:
            raise ValueError(f"stem_channels must be >= 1, got {self.stem_channels}")
        if any(c < 1 for c in self.conv_tower_channels):
            raise ValueError(f"conv_tower_channels must be positive, got {self.conv_tower_channels}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_attention_heads < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {self.num_attention_heads}")
        if self.target_length <= 0 and self.target_length != -1:
            raise ValueError(f"target_length must be positive (or -1 to skip cropping), got {self.target_length}")
        if self.pool_factor <= 0:
            raise ValueError(f"pool_factor must be positive, got {self.pool_factor}")
        if self.sequence_length % self.pool_factor != 0:
            raise ValueError(
                f"sequence_length ({self.sequence_length}) must be divisible by the total pooling factor "
                f"({self.pool_factor}) so the binned output is well defined."
            )
        if self.target_length != -1 and self.target_length > self.num_output_bins:
            raise ValueError(
                f"target_length ({self.target_length}) must not exceed the number of binned positions after "
                f"the U-Net upsampling tail ({self.num_output_bins})."
            )

    @property
    def num_downsamples(self) -> int:
        r"""Number of 2x downsampling stages: stem + tower + U-Net bottleneck + final pool."""
        # conv_dna pool (1) + one pool per reducing-tower stage except the last (len-1)
        # + unet1 pool (1) + final pool before transformer (1).
        return 1 + max(0, len(self.conv_tower_channels) - 1) + 2

    @property
    def num_upsamples(self) -> int:
        r"""Number of 2x upsampling stages in the U-Net tail."""
        return 2

    @property
    def pool_factor(self) -> int:
        r"""Total downsampling factor at the transformer trunk, i.e. base pairs per attention position."""
        return 2**self.num_downsamples

    @property
    def output_bin_size(self) -> int:
        r"""Base pairs per output bin, after the U-Net upsampling tail."""
        return 2 ** (self.num_downsamples - self.num_upsamples)

    @property
    def num_output_bins(self) -> int:
        r"""Number of binned positions produced by the U-Net upsampling tail before center-cropping."""
        return self.sequence_length // self.output_bin_size
