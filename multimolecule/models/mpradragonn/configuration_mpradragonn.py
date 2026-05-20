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


class MpraDragoNnConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`MpraDragoNnModel`][multimolecule.models.MpraDragoNnModel]. It is used to instantiate an MPRA-DragoNN model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MPRA-DragoNN
    [kundajelab/MPRA-DragoNN](https://github.com/kundajelab/MPRA-DragoNN) ConvModel architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the MPRA-DragoNN model. Defines the number of feature channels in the one-hot encoded
            input fed to the first convolution.
            Defaults to 5.
        input_length:
            The fixed length (in base pairs) of the input DNA sequence.
            Defaults to 145.
        num_conv_layers:
            Number of convolutional blocks (Conv1D + BatchNorm + activation + Dropout).
        conv_channels:
            Number of output channels for each convolutional block.
        conv_kernel_sizes:
            Convolution kernel size for each convolutional block.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability applied after each convolutional block.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers (PyTorch convention; equivalent to ``1 - momentum``
            in Keras, which uses 0.99 in the upstream checkpoint).
        num_labels:
            Number of regression outputs. MPRA-DragoNN predicts Sharpr-MPRA activity for 12 tasks: K562 / HepG2 cell
            lines, each with minP and SV40p reporter promoters, each measured as two replicates plus a pooled "avg"
            track (2 cells x 2 promoters x 3 measurements = 12 tasks).
        head:
            The configuration of the prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching MPRA-DragoNN's MPRA activity prediction task.

    Examples:
        >>> from multimolecule import MpraDragoNnConfig, MpraDragoNnModel
        >>> # Initializing an MPRA-DragoNN multimolecule/mpradragonn style configuration
        >>> configuration = MpraDragoNnConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/mpradragonn style configuration
        >>> model = MpraDragoNnModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "mpradragonn"

    def __init__(
        self,
        vocab_size: int = 5,
        input_length: int = 145,
        num_conv_layers: int = 3,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.1,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        num_labels: int = 12,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if conv_channels is None:
            conv_channels = [120, 120, 120]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [5, 5, 5]
        if len(conv_channels) != num_conv_layers:
            raise ValueError(f"conv_channels must have {num_conv_layers} entries, got {len(conv_channels)}.")
        if len(conv_kernel_sizes) != num_conv_layers:
            raise ValueError(f"conv_kernel_sizes must have {num_conv_layers} entries, got {len(conv_kernel_sizes)}.")
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}.")
        trimmed = input_length
        for kernel_size in conv_kernel_sizes:
            trimmed = trimmed - kernel_size + 1
            if trimmed <= 0:
                raise ValueError(
                    f"input_length={input_length} is too short for the configured conv stack; "
                    f"the feature map collapses after kernel sizes {conv_kernel_sizes}."
                )
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        # `hidden_size` is the dimensionality of the pooled feature vector that the prediction
        # head consumes (the flattened conv feature map), since MPRA-DragoNN has no learned pooler.
        self.hidden_size = trimmed * conv_channels[-1]
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

    @property
    def pooled_length(self) -> int:
        length = self.input_length
        for kernel_size in self.conv_kernel_sizes:
            length = length - kernel_size + 1
        return length
