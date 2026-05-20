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


class OptMrlConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`OptMrlModel`][multimolecule.models.OptMrlModel]. It is used to instantiate an OptMRL model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the OptMRL [ohlerlab/mlcis](https://github.com/ohlerlab/mlcis) architecture.

    OptMRL predicts the mean ribosome load (MRL) of an mRNA from the 50 nucleotides immediately upstream of the coding
    sequence. The published architecture is a three-layer 1D convolutional stack (same padding, length preserved)
    followed by a flattening dense bottleneck and a scalar regression head.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the OptMRL model. Defines the number of input channels of the first convolution.
            Defaults to 5 (`A`, `C`, `G`, `U`, `N`), matching the MultiMolecule RNA `streamline` alphabet. The
            upstream checkpoint only uses the first four (`A`, `C`, `G`, `U`); the `N` channel stays zero.
        sequence_length:
            The fixed 5'UTR input sequence length OptMRL was trained on (50 nt upstream of the coding sequence).
        num_conv_layers:
            Number of stacked 1D convolutions. The published OptMRL uses three.
        conv_filters:
            Number of filters in each convolutional layer.
        conv_kernel_size:
            Kernel size (sequence span) of each convolutional layer. Convolutions use `same` padding so the
            output length matches `sequence_length` after every layer.
        conv_dropout:
            Dropout probability applied after the second and third convolutions.
        dense_size:
            Number of units in the dense bottleneck consumed by the regression head.
        dense_dropout:
            Dropout probability applied after the dense bottleneck activation.
        hidden_act:
            The non-linear activation function used by the convolutional and dense layers.
        num_labels:
            Number of output labels. OptMRL is a single-output regression model, so this defaults to 1.
        head:
            The configuration of the sequence-level prediction head. Defaults to a regression head
            (`problem_type="regression"`).

    Examples:
        >>> from multimolecule import OptMrlConfig, OptMrlModel
        >>> # Initializing an OptMRL ohlerlab/mlcis style configuration
        >>> configuration = OptMrlConfig()
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = OptMrlModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "optmrl"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 50,
        num_conv_layers: int = 3,
        conv_filters: int = 120,
        conv_kernel_size: int = 8,
        conv_dropout: float = 0.0,
        dense_size: int = 40,
        dense_dropout: float = 0.2,
        hidden_act: str = "relu",
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if vocab_size < 4:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be at least 4 to cover the canonical nucleotide alphabet `ACGU`."
            )
        if sequence_length < 1:
            raise ValueError(f"sequence_length ({sequence_length}) must be a positive integer.")
        if conv_kernel_size < 1:
            raise ValueError(f"conv_kernel_size ({conv_kernel_size}) must be a positive integer.")
        if num_conv_layers < 1:
            raise ValueError(f"num_conv_layers ({num_conv_layers}) must be a positive integer.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_conv_layers = num_conv_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_dropout = conv_dropout
        self.dense_size = dense_size
        self.dense_dropout = dense_dropout
        self.hidden_act = hidden_act
        # ``hidden_size`` is the dimensionality of the dense bottleneck consumed by the
        # MultiMolecule sequence-prediction head.
        self.hidden_size = dense_size
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
