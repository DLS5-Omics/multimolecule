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


class DeepCpgDnaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DeepCpgDnaModel`][multimolecule.models.DeepCpgDnaModel]. It is used to instantiate a DeepCpG-DNA model according
    to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the DeepCpG DNA submodule
    ([cangermueller/deepcpg](https://github.com/cangermueller/deepcpg)) `CnnL2h128` architecture as distributed for the
    Smallwood2014 serum mESC checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the DeepCpG-DNA model. DeepCpG consumes a one-hot encoding of DNA nucleotides, so this
            also defines the number of input channels of the first convolution. Defaults to 5 to match the
            MultiMolecule `streamline` DNA alphabet (`A`, `C`, `G`, `T`, `N`); the upstream four-channel kernel is
            reordered into this slot layout in the converter, leaving the `N` channel zero.
            Defaults to 5.
        sequence_length:
            The fixed length of the DNA window (in base pairs) centered on a CpG site.
            Defaults to 1001.
        conv_channels:
            Number of filters for each convolutional layer.
        conv_kernel_sizes:
            Kernel size for each convolutional layer.
        conv_pool_sizes:
            Max-pool size applied after each convolutional layer.
        bottleneck_size:
            Dimensionality of the dense bottleneck embedding. This is the model's hidden size.
            Defaults to 128.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability for the bottleneck.
        num_labels:
            Number of output labels. DeepCpG-DNA predicts per-cell methylation state, so this equals the number of
            single cells in the training dataset and is **dataset-specific**.
            Defaults to 18 to match the Smallwood2014 serum mESC checkpoint.
        head:
            The configuration of the prediction head. Defaults to a per-cell binary methylation head
            (`problem_type="binary"`), matching DeepCpG-DNA's per-cell methylation task.

    Examples:
        >>> from multimolecule import DeepCpgDnaConfig, DeepCpgDnaModel
        >>> # Initializing a DeepCpG-DNA multimolecule/deepcpgdna style configuration
        >>> configuration = DeepCpgDnaConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/deepcpgdna style configuration
        >>> model = DeepCpgDnaModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "deepcpgdna"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 1001,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        conv_pool_sizes: list[int] | None = None,
        bottleneck_size: int = 128,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
        num_labels: int = 18,
        head: HeadConfig | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, **kwargs)
        # Upstream `CnnL2h128`: Conv1D(128, 11) -> MaxPool(4) -> Conv1D(256, 3) -> MaxPool(2) -> Flatten -> Dense(128).
        if conv_channels is None:
            conv_channels = [128, 256]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [11, 3]
        if conv_pool_sizes is None:
            conv_pool_sizes = [4, 2]
        if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_pool_sizes)):
            raise ValueError(
                "conv_channels, conv_kernel_sizes and conv_pool_sizes must have the same length, but got "
                f"{len(conv_channels)}, {len(conv_kernel_sizes)} and {len(conv_pool_sizes)}."
            )
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, but got {sequence_length}.")
        if bottleneck_size <= 0:
            raise ValueError(f"bottleneck_size must be positive, but got {bottleneck_size}.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pool_sizes = conv_pool_sizes
        self.bottleneck_size = bottleneck_size
        self.hidden_size = bottleneck_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        # DeepCpG-DNA performs per-cell binary methylation prediction. The MultiMolecule `problem_type` convention
        # lives on the head config, since the Transformers base config only accepts the HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="binary")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "binary"
        self.head = head
