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


class DeepStarrConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DeepStarrModel`][multimolecule.models.DeepStarrModel]. It is used to instantiate a DeepSTARR model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the DeepSTARR
    [bernardo-de-almeida/DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the DeepSTARR model. Defines the number of feature channels in the one-hot encoded
            input fed to the first convolution.
            Defaults to 5.
        input_length:
            The fixed length (in base pairs) of the input DNA sequence.
            Defaults to 249.
        num_conv_layers:
            Number of convolutional blocks (Conv1D + BatchNorm + ReLU + MaxPool).
        conv_channels:
            Number of output channels for each convolutional block.
        conv_kernel_sizes:
            Convolution kernel size for each convolutional block.
        pool_size:
            Max pooling window applied after every convolutional block.
        num_fc_layers:
            Number of fully-connected layers between the convolutional stack and the prediction head.
        fc_dims:
            Hidden size for each fully-connected layer.
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
            Number of regression outputs. DeepSTARR predicts developmental and housekeeping enhancer activity.
        head:
            The configuration of the prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching DeepSTARR's enhancer activity prediction task.

    Examples:
        >>> from multimolecule import DeepStarrConfig, DeepStarrModel
        >>> # Initializing a DeepSTARR multimolecule/deepstarr style configuration
        >>> configuration = DeepStarrConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/deepstarr style configuration
        >>> model = DeepStarrModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "deepstarr"

    def __init__(
        self,
        vocab_size: int = 5,
        input_length: int = 249,
        num_conv_layers: int = 4,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        pool_size: int = 2,
        num_fc_layers: int = 2,
        fc_dims: list[int] | None = None,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.4,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 2,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if conv_channels is None:
            conv_channels = [256, 60, 60, 120]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [7, 3, 5, 3]
        if fc_dims is None:
            fc_dims = [256, 256]
        if len(conv_channels) != num_conv_layers:
            raise ValueError(f"conv_channels must have {num_conv_layers} entries, got {len(conv_channels)}.")
        if len(conv_kernel_sizes) != num_conv_layers:
            raise ValueError(f"conv_kernel_sizes must have {num_conv_layers} entries, got {len(conv_kernel_sizes)}.")
        if len(fc_dims) != num_fc_layers:
            raise ValueError(f"fc_dims must have {num_fc_layers} entries, got {len(fc_dims)}.")
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}.")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}.")
        if not fc_dims:
            raise ValueError("fc_dims must contain at least one fully-connected layer.")
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_size = pool_size
        self.num_fc_layers = num_fc_layers
        self.fc_dims = fc_dims
        self.hidden_size = fc_dims[-1]
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
