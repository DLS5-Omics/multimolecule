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


class ScBassetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`ScBassetModel`][multimolecule.models.ScBassetModel]. It is used to instantiate a scBasset model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the scBasset [calico/scBasset](https://github.com/calico/scBasset) architecture as
    distributed for the Buenrostro2018 hematopoiesis tutorial checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the scBasset model. scBasset consumes a one-hot encoding of DNA nucleotides, so this
            also defines the number of input channels of the first convolution. Defaults to 5 to match the
            MultiMolecule `streamline` DNA alphabet (`A`, `C`, `G`, `T`, `N`); the upstream four-channel kernel is
            reordered into this slot layout in the converter, leaving the `N` channel zero.
            Defaults to 5.
        sequence_length:
            The fixed length of the input DNA peak sequence in base pairs.
            Defaults to 1344.
        stem_channels:
            Number of filters in the stem (first) convolution.
            Defaults to 288.
        stem_kernel_size:
            Kernel size of the stem convolution.
            Defaults to 17.
        stem_pool_size:
            Max-pool size applied after the stem convolution.
            Defaults to 3.
        tower_channels:
            Number of filters for each convolution in the reducing tower. The upstream architecture derives these
            from `filters_init=288`, `filters_mult=1.122`, `repeat=6` with integer rounding.
        tower_kernel_size:
            Kernel size for each tower convolution.
            Defaults to 5.
        tower_pool_size:
            Max-pool size applied after each tower convolution.
            Defaults to 2.
        pointwise_channels:
            Number of filters in the final pointwise (kernel size 1) convolution.
            Defaults to 256.
        bottleneck_size:
            Dimensionality of the dense bottleneck embedding. This is the model's hidden size.
            Defaults to 32.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. scBasset uses the sigmoid
            approximation of GELU (`sigmoid(1.702 * x) * x`), exposed by Transformers as `"quick_gelu"`.
        hidden_dropout:
            The dropout probability for the bottleneck.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of output labels. scBasset predicts per-cell chromatin accessibility, so this equals the number
            of single cells in the training atlas and is **dataset-specific**. The shipped Buenrostro2018
            hematopoiesis checkpoint has 2034 cells.
            Defaults to 2034.
        head:
            The configuration of the prediction head. Defaults to a per-cell binary accessibility head
            (`problem_type="binary"`), matching scBasset's per-cell accessibility task.

    Examples:
        >>> from multimolecule import ScBassetConfig, ScBassetModel
        >>> # Initializing a scBasset multimolecule/scbasset style configuration
        >>> configuration = ScBassetConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/scbasset style configuration
        >>> model = ScBassetModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "scbasset"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 1344,
        stem_channels: int = 288,
        stem_kernel_size: int = 17,
        stem_pool_size: int = 3,
        tower_channels: list[int] | None = None,
        tower_kernel_size: int = 5,
        tower_pool_size: int = 2,
        pointwise_channels: int = 256,
        bottleneck_size: int = 32,
        hidden_act: str = "quick_gelu",
        hidden_dropout: float = 0.2,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 2034,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if tower_channels is None:
            # Upstream conv_tower(filters_init=288, filters_mult=1.122, repeat=6) with int(round(...)) rounding.
            tower_channels = [288, 323, 363, 407, 456, 512]
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, but got {sequence_length}.")
        if bottleneck_size <= 0:
            raise ValueError(f"bottleneck_size must be positive, but got {bottleneck_size}.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.stem_channels = stem_channels
        self.stem_kernel_size = stem_kernel_size
        self.stem_pool_size = stem_pool_size
        self.tower_channels = tower_channels
        self.tower_kernel_size = tower_kernel_size
        self.tower_pool_size = tower_pool_size
        self.pointwise_channels = pointwise_channels
        self.bottleneck_size = bottleneck_size
        self.hidden_size = bottleneck_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        # scBasset performs per-cell binary accessibility prediction. The MultiMolecule `problem_type` convention
        # lives on the head config, since the Transformers base config only accepts the HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="binary")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "binary"
        self.head = head
