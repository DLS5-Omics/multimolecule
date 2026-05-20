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

DEFAULT_POSITION_FEATURES = (
    "tss",
    "polya",
    "exon_intron",
    "intron_exon",
    "start_codon",
    "stop_codon",
    "gene_start",
    "gene_end",
)


class RbpEclipConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`RbpEclipModel`][multimolecule.models.RbpEclipModel]. It is used to instantiate an RBP-eCLIP model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the per-RBP eCLIP models distributed via Kipoi
    ([`rbp_eclip`](https://kipoi.org/models/rbp_eclip/)), trained with the architecture from
    [Avsec et al. Bioinformatics 2018](https://doi.org/10.1093/bioinformatics/btx727).

    The model is a small 1D convolutional network that combines an RNA sequence module with an optional
    position module. The position module evaluates eight genomic-landmark distance features through a
    pre-computed B-spline basis (`num_spline_bases`) and a generalised additive (GAM) 1x1 convolution.
    Each trained Hub checkpoint corresponds to a single RBP.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the RBP-eCLIP model. Defines the number of input channels of the first convolution.
            The upstream Kipoi checkpoints one-hot encode RNA as `["A", "C", "G", "U"]`; the MultiMolecule
            `streamline` RNA alphabet adds an `N` channel that stays zero for in-vocabulary inputs.
        sequence_length:
            The fixed length of the input RNA peak window in nucleotides. Defaults to 101, the window used by
            the published eCLIP RBP models.
        num_sequence_filters:
            Number of filters in both convolutional layers of the sequence module. Matches the upstream
            `filters` hyper-parameter.
        sequence_kernel_size:
            Kernel size of the first sequence convolution. The second sequence convolution always uses a 1x1
            kernel. Matches the upstream `kernel_size` hyper-parameter.
        sequence_pool_size:
            Stride / pool window applied to the sequence-module output before flattening. Matches the
            upstream `internal_pos = {"name": "strided_maxpool", "pool_size": ...}` configuration.
        hidden_act:
            The non-linear activation function (function or string) used by the convolutional and dense layers.
            If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        conv2_use_skip:
            If True, the second sequence convolution output is concatenated with the first sequence convolution
            output before the strided pooling. Matches the upstream `conv2_use_skip` hyper-parameter.
        use_batchnorm:
            If True, batch normalization is applied after each convolutional layer (over the sequence-length
            axis, axis=1, matching the upstream `kl.BatchNormalization(axis=1)`) and after the pooled features
            and the hidden dense layer (over the feature axis). Matches the upstream `use_batchnorm`
            hyper-parameter.
        batch_norm_eps:
            Epsilon added to the variance in batch normalization. The Kipoi `rbp_eclip` checkpoints inherit
            Keras's default `epsilon=1e-3`, which is preserved here so the converted PyTorch BN reproduces
            the upstream numerics exactly.
        batch_norm_momentum:
            Momentum used to update the batch-normalization running statistics. Matches the upstream
            Keras default of 0.99 (PyTorch convention `1 - momentum`, i.e. 0.01 in `nn.BatchNorm1d`).
        num_hidden:
            Number of hidden units in the dense layer that consumes the pooled sequence features and the
            position-module scalars.
        hidden_dropout:
            Dropout probability applied after the pooled sequence features and after the dense layer.
        num_position_features:
            Number of genomic-landmark distance features consumed by the position module. The published
            architecture uses eight landmarks (TSS, poly-A, exon-intron, intron-exon, start codon, stop codon,
            gene start, gene end).
        num_position_filters:
            Number of filters of each position-module GAM 1x1 convolution. Matches the upstream `units` field
            of `external_pos`. The Kipoi `rbp_eclip` checkpoints use `units=1`.
        num_spline_bases:
            Number of B-spline basis functions used to pre-compute each distance feature. Matches the
            upstream `n_bases` field. The default of 10 matches the Kipoi `rbp_eclip` checkpoints.
        position_feature_names:
            Optional ordered list of position-feature names. Used to label outputs and to derive defaults for
            `num_position_features`. Defaults to the eight upstream landmarks.
        num_labels:
            Number of output labels per sequence. The published eCLIP RBP models predict a single binding
            score per sequence, so this defaults to 1.
        head:
            The configuration of the sequence-level prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching the per-RBP binding-score output.

    Examples:
        >>> from multimolecule import RbpEclipConfig, RbpEclipModel
        >>> # Initializing an RBP-eCLIP multimolecule/rbpeclip style configuration
        >>> configuration = RbpEclipConfig()
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = RbpEclipModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "rbpeclip"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 101,
        num_sequence_filters: int = 16,
        sequence_kernel_size: int = 11,
        sequence_pool_size: int = 4,
        hidden_act: str = "relu",
        conv2_use_skip: bool = False,
        use_batchnorm: bool = True,
        num_hidden: int = 100,
        hidden_dropout: float = 0.5,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        num_position_features: int = 8,
        num_position_filters: int = 1,
        num_spline_bases: int = 10,
        position_feature_names: list[str] | None = None,
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if position_feature_names is None:
            position_feature_names = list(DEFAULT_POSITION_FEATURES)
        if num_position_features != len(position_feature_names):
            raise ValueError(
                f"num_position_features ({num_position_features}) must equal len(position_feature_names) "
                f"({len(position_feature_names)})."
            )
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, but got {sequence_length}.")
        if sequence_kernel_size <= 0 or sequence_kernel_size > sequence_length:
            raise ValueError(
                f"sequence_kernel_size ({sequence_kernel_size}) must be in (0, sequence_length={sequence_length}]."
            )
        if sequence_pool_size <= 0:
            raise ValueError(f"sequence_pool_size must be positive, but got {sequence_pool_size}.")
        if num_spline_bases <= 0:
            raise ValueError(f"num_spline_bases must be positive, but got {num_spline_bases}.")
        if num_position_filters <= 0:
            raise ValueError(f"num_position_filters must be positive, but got {num_position_filters}.")
        if num_hidden <= 0:
            raise ValueError(f"num_hidden must be positive, but got {num_hidden}.")
        if num_sequence_filters <= 0:
            raise ValueError(f"num_sequence_filters must be positive, but got {num_sequence_filters}.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_sequence_filters = num_sequence_filters
        self.sequence_kernel_size = sequence_kernel_size
        self.sequence_pool_size = sequence_pool_size
        self.hidden_act = hidden_act
        self.conv2_use_skip = conv2_use_skip
        self.use_batchnorm = use_batchnorm
        self.num_hidden = num_hidden
        self.hidden_dropout = hidden_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.num_position_features = num_position_features
        self.num_position_filters = num_position_filters
        self.num_spline_bases = num_spline_bases
        self.position_feature_names = list(position_feature_names)
        # ``hidden_size`` is the dimensionality of the shared dense representation consumed by the
        # MultiMolecule sequence-prediction head (the output of the single hidden FC layer).
        self.hidden_size = num_hidden
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head

    @property
    def post_conv1_length(self) -> int:
        r"""Sequence length after the first sequence convolution (`valid` padding)."""
        return self.sequence_length - self.sequence_kernel_size + 1

    @property
    def pooled_length(self) -> int:
        r"""Sequence length after strided max-pool, matching Keras `MaxPool1D` (floor division)."""
        return self.post_conv1_length // self.sequence_pool_size

    @property
    def pooled_features(self) -> int:
        r"""Flattened sequence-feature size consumed by the hidden dense layer."""
        skip_factor = 2 if self.conv2_use_skip else 1
        return self.pooled_length * self.num_sequence_filters * skip_factor
