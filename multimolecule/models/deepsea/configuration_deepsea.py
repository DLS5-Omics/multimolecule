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


class DeepSeaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DeepSeaModel`][multimolecule.models.DeepSeaModel]. It is used to instantiate a DeepSEA model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the DeepSEA architecture from Zhou & Troyanskaya (Nat. Methods 2015).

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the DeepSEA model. DeepSEA consumes a one-hot encoding of the four DNA nucleotides, so
            this also defines the number of input channels of the first convolution.
            Defaults to 4.
        sequence_length:
            The fixed length of the input DNA sequence in base pairs.
            Defaults to 1000.
        num_conv_layers:
            Number of convolutional layers in the encoder.
        conv_channels:
            Number of filters for each convolutional layer.
        conv_kernel_sizes:
            Kernel size for each convolutional layer.
        conv_pool_sizes:
            Max-pool size applied after each convolutional layer. A value of `1` means no pooling is applied after
            that layer (DeepSEA omits the pool between the third convolution and the fully-connected stack).
        conv_dropouts:
            Dropout probability applied after each convolutional layer.
        fc_sizes:
            Hidden dimensionality of each fully-connected layer between the convolutional stack and the output head.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability applied between the fully-connected layer and the output head.
        reverse_complement_average:
            Whether [`DeepSeaForSequencePrediction`][multimolecule.models.DeepSeaForSequencePrediction] averages
            forward and reverse-complement prediction probabilities.
            Defaults to True, matching the DeepSEA sequence-prediction checkpoint.
        num_labels:
            Number of output labels. DeepSEA predicts 919 chromatin-feature probabilities (DNase I hypersensitivity,
            transcription-factor binding, and histone-mark peaks across multiple cell types).
            Defaults to 919.
        head:
            The configuration of the prediction head. Defaults to a multi-label binary classification head
            (`problem_type="multilabel"`), matching DeepSEA's chromatin-feature prediction task.

    Examples:
        >>> from multimolecule import DeepSeaConfig, DeepSeaModel
        >>> # Initializing a DeepSEA multimolecule/deepsea style configuration
        >>> configuration = DeepSeaConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/deepsea style configuration
        >>> model = DeepSeaModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "deepsea"

    def __init__(
        self,
        vocab_size: int = 4,
        sequence_length: int = 1000,
        num_conv_layers: int = 3,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        conv_pool_sizes: list[int] | None = None,
        conv_dropouts: list[float] | None = None,
        fc_sizes: list[int] | None = None,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
        reverse_complement_average: bool = True,
        num_labels: int = 919,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if conv_channels is None:
            conv_channels = [320, 480, 960]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [8, 8, 8]
        if conv_pool_sizes is None:
            # Upstream DeepSEA pools after the first two convolutions only; the third convolution
            # is followed directly by the heavy 0.5 dropout and the fully-connected classifier.
            conv_pool_sizes = [4, 4, 1]
        if conv_dropouts is None:
            conv_dropouts = [0.2, 0.2, 0.5]
        if fc_sizes is None:
            fc_sizes = [925]
        lengths = (len(conv_channels), len(conv_kernel_sizes), len(conv_pool_sizes), len(conv_dropouts))
        if any(length != num_conv_layers for length in lengths):
            raise ValueError(
                "conv_channels, conv_kernel_sizes, conv_pool_sizes and conv_dropouts must each have length "
                f"num_conv_layers ({num_conv_layers}), but got {lengths[0]}, {lengths[1]}, {lengths[2]} "
                f"and {lengths[3]}."
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
        self.conv_dropouts = conv_dropouts
        self.fc_sizes = fc_sizes
        self.hidden_size = fc_sizes[-1]
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.reverse_complement_average = reverse_complement_average
        # DeepSEA performs multi-label binary classification of 919 chromatin features. The MultiMolecule
        # `problem_type` convention lives on the head config, since the Transformers base config only accepts
        # the HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="multilabel")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "multilabel"
        self.head = head
