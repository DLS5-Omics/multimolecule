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


class XpressoConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`XpressoModel`][multimolecule.models.XpressoModel]. It is used to instantiate a Xpresso model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the Xpresso
    [vagarwal87/Xpresso](https://github.com/vagarwal87/Xpresso) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the Xpresso model. Defines the number of feature channels derived from `input_ids` for
            the first convolution. Defaults to 5.
        input_length:
            The length of the promoter sequence window (centered on the TSS) consumed by the convolutional stack.
        num_conv_layers:
            Number of convolutional blocks in the encoder.
        conv_channels:
            Number of output channels for each convolutional block. Length must equal `num_conv_layers`.
        conv_kernel_sizes:
            Convolution kernel size for each convolutional block. Length must equal `num_conv_layers`.
        conv_dilations:
            Dilation factor for each convolutional block. Length must equal `num_conv_layers`.
        pool_sizes:
            Max-pooling window for each convolutional block. Length must equal `num_conv_layers`.
        num_features:
            Number of auxiliary numeric mRNA half-life features concatenated with the convolutional representation
            before the fully-connected head.
        fc_dims:
            Dimensionality of each fully-connected layer in the head.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and the head. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability applied after each fully-connected layer.
        num_labels:
            Number of output labels. Xpresso predicts a single scalar mRNA expression value.
        head:
            The configuration of the prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching Xpresso's mRNA abundance prediction task.

    Examples:
        >>> from multimolecule import XpressoConfig, XpressoModel
        >>> # Initializing a Xpresso multimolecule/xpresso style configuration
        >>> configuration = XpressoConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/xpresso style configuration
        >>> model = XpressoModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "xpresso"

    def __init__(
        self,
        vocab_size: int = 5,
        input_length: int = 10500,
        num_conv_layers: int = 2,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        conv_dilations: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        num_features: int = 6,
        fc_dims: list[int] | None = None,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.00099,
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        kwargs.setdefault("pad_token_id", vocab_size - 1)
        kwargs.setdefault("unk_token_id", vocab_size - 1)
        kwargs.setdefault("bos_token_id", None)
        kwargs.setdefault("eos_token_id", None)
        kwargs.setdefault("mask_token_id", None)
        kwargs.setdefault("null_token_id", None)
        super().__init__(num_labels=num_labels, **kwargs)
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.num_conv_layers = num_conv_layers
        if conv_channels is None:
            conv_channels = [128, 32]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [6, 9]
        if conv_dilations is None:
            conv_dilations = [1, 1]
        if pool_sizes is None:
            pool_sizes = [30, 10]
        if fc_dims is None:
            fc_dims = [64, 2]
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_dilations = conv_dilations
        self.pool_sizes = pool_sizes
        self.num_features = num_features
        self.fc_dims = fc_dims
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.num_labels = num_labels
        # `hidden_size` is the dimensionality of the pooled representation consumed by
        # `SequencePredictionHead`; it equals the width of the last fully-connected layer.
        self.hidden_size = self.fc_dims[-1]
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
        self._validate()

    def _validate(self) -> None:
        per_layer = {
            "conv_channels": self.conv_channels,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "conv_dilations": self.conv_dilations,
            "pool_sizes": self.pool_sizes,
        }
        for name, value in per_layer.items():
            if len(value) != self.num_conv_layers:
                raise ValueError(
                    f"`{name}` must have length `num_conv_layers` ({self.num_conv_layers}), got {len(value)}."
                )
        if self.input_length <= 0:
            raise ValueError(f"`input_length` must be positive, got {self.input_length}.")
        if self.num_features < 0:
            raise ValueError(f"`num_features` must be non-negative, got {self.num_features}.")
        if not self.fc_dims:
            raise ValueError("`fc_dims` must contain at least one fully-connected dimension.")
