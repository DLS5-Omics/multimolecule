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


class MalinoisConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`MalinoisModel`][multimolecule.models.MalinoisModel]. It is used to instantiate a Malinois model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the Malinois [sjgosai/boda2](https://github.com/sjgosai/boda2) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the Malinois model. Defines the number of feature channels in the one-hot encoded
            input fed to the first convolution.
            Defaults to 5.
        input_length:
            The fixed length (in base pairs) of the input fed to the first convolution. Upstream Malinois pads each
            200 bp candidate sequence with fixed MPRA plasmid flanks up to this length before the convolution stack.
            Defaults to 600.
        conv_channels:
            Number of output channels for each convolutional block.
        conv_kernel_sizes:
            Convolution kernel size for each convolutional block.
        num_linear_layers:
            Number of fully-connected layers between the convolutional stack and the branched tower.
        linear_channels:
            Hidden size for each fully-connected layer.
        linear_act:
            The non-linear activation function (function or string) applied after the convolutional and linear
            layers. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        linear_dropout:
            The dropout probability for the fully-connected layers.
        num_branched_layers:
            Number of grouped (branched) layers, one independent tower per output cell line.
        branched_channels:
            Hidden size for each branch in the branched tower.
        branched_act:
            The non-linear activation function applied between branched layers.
        branched_dropout:
            The dropout probability for the branched tower.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of regression outputs. Malinois predicts cell-type-informed cis-regulatory activity for three
            human cell lines: K562, HepG2 and SK-N-SH (in that order).
        head:
            The configuration of the prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching Malinois's CRE activity prediction task.

    Examples:
        >>> from multimolecule import MalinoisConfig, MalinoisModel
        >>> # Initializing a Malinois multimolecule/malinois style configuration
        >>> configuration = MalinoisConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/malinois style configuration
        >>> model = MalinoisModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "malinois"

    def __init__(
        self,
        vocab_size: int = 5,
        input_length: int = 600,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        num_linear_layers: int = 1,
        linear_channels: int = 1000,
        linear_act: str = "relu",
        linear_dropout: float = 0.11625456877954289,
        num_branched_layers: int = 3,
        branched_channels: int = 140,
        branched_act: str = "relu",
        branched_dropout: float = 0.5757068086404574,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 3,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if conv_channels is None:
            conv_channels = [300, 200, 200]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [19, 11, 7]
        if len(conv_channels) != len(conv_kernel_sizes):
            raise ValueError(
                f"conv_channels and conv_kernel_sizes must have the same length, "
                f"got {len(conv_channels)} and {len(conv_kernel_sizes)}."
            )
        if len(conv_channels) != 3:
            raise ValueError(f"Malinois uses exactly 3 convolutional blocks, got {len(conv_channels)}.")
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}.")
        if num_linear_layers <= 0:
            raise ValueError(f"num_linear_layers must be positive, got {num_linear_layers}.")
        if num_branched_layers <= 0:
            raise ValueError(f"num_branched_layers must be positive, got {num_branched_layers}.")
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_conv_layers = len(conv_channels)
        self.num_linear_layers = num_linear_layers
        self.linear_channels = linear_channels
        self.linear_act = linear_act
        self.linear_dropout = linear_dropout
        self.num_branched_layers = num_branched_layers
        self.branched_channels = branched_channels
        self.branched_act = branched_act
        self.branched_dropout = branched_dropout
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
    def flatten_factor(self) -> int:
        hook = self.input_length // 3 // 4
        return (hook + 2) // 4

    @property
    def hidden_size(self) -> int:
        return self.num_labels * self.branched_channels
