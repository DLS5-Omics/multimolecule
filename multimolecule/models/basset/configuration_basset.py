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


class BassetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`BassetModel`][multimolecule.models.BassetModel]. It is used to instantiate a Basset model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the Basset [davek44/Basset](https://github.com/davek44/Basset) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the Basset model. Basset consumes a one-hot encoding of the four DNA nucleotides, so
            this also defines the number of input channels of the first convolution.
            Defaults to 4.
        sequence_length:
            The fixed length of the input DNA sequence in base pairs.
            Defaults to 600.
        num_conv_layers:
            Number of convolutional layers in the encoder.
        conv_channels:
            Number of filters for each convolutional layer.
        conv_kernel_sizes:
            Kernel size for each convolutional layer.
        conv_pool_sizes:
            Max-pool size applied after each convolutional layer.
        fc_sizes:
            Hidden dimensionality of each fully-connected layer.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability for the fully-connected layers.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of output labels. Basset predicts DNase I hypersensitivity across 164 cell types.
            Defaults to 164.
        head:
            The configuration of the prediction head. Defaults to a multi-label binary classification head
            (`problem_type="multilabel"`), matching Basset's DNase I hypersensitivity prediction task.

    Examples:
        >>> from multimolecule import BassetConfig, BassetModel
        >>> # Initializing a Basset multimolecule/basset style configuration
        >>> configuration = BassetConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/basset style configuration
        >>> model = BassetModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "basset"

    def __init__(
        self,
        vocab_size: int = 4,
        sequence_length: int = 600,
        num_conv_layers: int = 3,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        conv_pool_sizes: list[int] | None = None,
        fc_sizes: list[int] | None = None,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.3,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 164,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if conv_channels is None:
            conv_channels = [300, 200, 200]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [19, 11, 7]
        if conv_pool_sizes is None:
            conv_pool_sizes = [3, 4, 4]
        if fc_sizes is None:
            fc_sizes = [1000, 1000]
        if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_pool_sizes) == num_conv_layers):
            raise ValueError(
                "conv_channels, conv_kernel_sizes and conv_pool_sizes must each have length num_conv_layers "
                f"({num_conv_layers}), but got {len(conv_channels)}, {len(conv_kernel_sizes)} and "
                f"{len(conv_pool_sizes)}."
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
        self.fc_sizes = fc_sizes
        self.hidden_size = fc_sizes[-1]
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        # Basset performs multi-label binary classification of DNase I hypersensitivity. The MultiMolecule
        # `problem_type` convention lives on the head config, since the Transformers base config only accepts
        # the HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="multilabel")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "multilabel"
        self.head = head
