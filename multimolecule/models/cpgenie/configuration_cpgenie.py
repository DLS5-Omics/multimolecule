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


class CpGenieConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`CpGenieModel`][multimolecule.models.CpGenieModel]. It is used to instantiate a CpGenie model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the CpGenie [gifford-lab/CpGenie](https://github.com/gifford-lab/CpGenie)
    `seq_128x3_5_5_2f_simple` architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the CpGenie model. CpGenie consumes a one-hot encoding of the four DNA nucleotides, so
            this also defines the number of input channels of the first convolution.
            Defaults to 4.
        sequence_length:
            The fixed length of the input DNA window in base pairs. CpGenie predicts the methylation state of the
            central CpG dinucleotide; the 501st nucleotide is expected to be the `C` of the CpG.
            Defaults to 1001.
        num_conv_layers:
            Number of convolutional layers in the encoder.
        conv_channels:
            Number of filters for each convolutional layer.
        conv_kernel_sizes:
            Kernel size for each convolutional layer.
        conv_pool_sizes:
            Max-pool window applied after each convolutional layer.
        conv_pool_strides:
            Max-pool stride applied after each convolutional layer.
        fc_sizes:
            Hidden dimensionality of each fully-connected layer that follows the convolutional stack.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability applied between the fully-connected layers.
        num_labels:
            Number of output labels. Each CpGenie model is trained as a 2-class softmax classifier (unmethylated vs.
            methylated) for the central CpG of a single ENCODE RRBS cell line.
            Defaults to 2.
        head:
            The configuration of the prediction head. Defaults to a 2-class softmax classification head
            (`problem_type="multiclass"`), matching CpGenie's per-cell-line methylation task.

    Examples:
        >>> from multimolecule import CpGenieConfig, CpGenieModel
        >>> # Initializing a CpGenie multimolecule/cpgenie style configuration
        >>> configuration = CpGenieConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/cpgenie style configuration
        >>> model = CpGenieModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "cpgenie"

    def __init__(
        self,
        vocab_size: int = 4,
        sequence_length: int = 1001,
        num_conv_layers: int = 3,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        conv_pool_sizes: list[int] | None = None,
        conv_pool_strides: list[int] | None = None,
        fc_sizes: list[int] | None = None,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.5,
        num_labels: int = 2,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if conv_channels is None:
            conv_channels = [128, 256, 512]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [5, 5, 5]
        if conv_pool_sizes is None:
            conv_pool_sizes = [5, 5, 5]
        if conv_pool_strides is None:
            conv_pool_strides = [3, 3, 3]
        if fc_sizes is None:
            fc_sizes = [64, 64]
        expected = num_conv_layers
        if not (
            len(conv_channels) == len(conv_kernel_sizes) == len(conv_pool_sizes) == len(conv_pool_strides) == expected
        ):
            raise ValueError(
                "conv_channels, conv_kernel_sizes, conv_pool_sizes and conv_pool_strides must each have length "
                f"num_conv_layers ({num_conv_layers}), but got {len(conv_channels)}, {len(conv_kernel_sizes)}, "
                f"{len(conv_pool_sizes)} and {len(conv_pool_strides)}."
            )
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, but got {sequence_length}.")
        if not fc_sizes:
            raise ValueError("fc_sizes must contain at least one fully-connected layer.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pool_sizes = conv_pool_sizes
        self.conv_pool_strides = conv_pool_strides
        self.fc_sizes = fc_sizes
        self.hidden_size = fc_sizes[-1]
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        # CpGenie performs per-cell-line methylation classification as a 2-class softmax problem
        # (unmethylated vs. methylated). The MultiMolecule `problem_type` convention lives on the head
        # config, since the Transformers base config only accepts the HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="multiclass")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "multiclass"
        self.head = head
