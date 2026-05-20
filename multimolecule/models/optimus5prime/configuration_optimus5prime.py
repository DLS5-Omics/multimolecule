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


class Optimus5PrimeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`Optimus5PrimeModel`][multimolecule.models.Optimus5PrimeModel]. It is used to instantiate an Optimus 5-Prime model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Optimus 5-Prime main MRL model from
    [pjsample/human_5utr_modeling](https://github.com/pjsample/human_5utr_modeling).

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the Optimus 5-Prime model. Defines the number of one-hot input channels derived from
            `input_ids`. Defaults to 5 (the MultiMolecule RNA `streamline` alphabet `ACGUN`); the upstream checkpoint
            only uses the first four (`A`, `C`, `G`, `U`/`T`) and the `N` channel stays zero.
        sequence_length:
            The fixed 5'UTR input sequence length Optimus 5-Prime was trained on (50 nt).
        num_conv_layers:
            Number of stacked 1D convolutions. The published main MRL model uses 3.
        conv_channels:
            Number of output channels in every convolution. The published main MRL model uses 120.
        conv_kernel_size:
            Convolution kernel size. The published main MRL model uses 8 with `padding="same"`.
        conv_dropout:
            Dropout probability applied after each intermediate convolution. The published main MRL model uses 0.0.
        hidden_size:
            Size of the fully connected layer between the convolutional stack and the regression output. The published
            main MRL model uses 40.
        dense_dropout:
            Dropout probability applied after the dense hidden layer. The published main MRL model uses 0.2.
        hidden_act:
            The non-linear activation function used by the convolutional and dense layers.
        num_labels:
            Number of output labels. Optimus 5-Prime predicts a single mean ribosome load (MRL) scalar, so this
            defaults to 1.
        head:
            The configuration of the sequence-level prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching Optimus 5-Prime's MRL regression task.

    Examples:
        >>> from multimolecule import Optimus5PrimeConfig, Optimus5PrimeModel
        >>> # Initializing an Optimus 5-Prime style configuration
        >>> configuration = Optimus5PrimeConfig()
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = Optimus5PrimeModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "optimus5prime"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 50,
        num_conv_layers: int = 3,
        conv_channels: int = 120,
        conv_kernel_size: int = 8,
        conv_dropout: float = 0.0,
        hidden_size: int = 40,
        dense_dropout: float = 0.2,
        hidden_act: str = "relu",
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if vocab_size < 4:
            raise ValueError(
                f"vocab_size ({vocab_size}) must cover the four canonical nucleotides used by Optimus 5-Prime."
            )
        if sequence_length <= 0:
            raise ValueError(f"sequence_length ({sequence_length}) must be a positive integer.")
        if num_conv_layers < 1:
            raise ValueError(f"num_conv_layers ({num_conv_layers}) must be >= 1.")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels ({conv_channels}) must be positive.")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size ({conv_kernel_size}) must be positive.")
        if not 0.0 <= conv_dropout < 1.0:
            raise ValueError(f"conv_dropout ({conv_dropout}) must be in [0.0, 1.0).")
        if not 0.0 <= dense_dropout < 1.0:
            raise ValueError(f"dense_dropout ({dense_dropout}) must be in [0.0, 1.0).")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be positive.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_dropout = conv_dropout
        self.hidden_size = hidden_size
        self.dense_dropout = dense_dropout
        self.hidden_act = hidden_act
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
