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

from chanfig import FlatDict

from ..configuration_utils import PreTrainedConfig


class SpliceAiStageConfig(FlatDict):
    r"""
    Configuration for a single SpliceAI stage.

    Args:
        num_blocks:
            Number of convolutional blocks in the stage.
        kernel_size:
            Convolution kernel size for the stage.
        dilation:
            Dilation factor for the stage.
    """

    num_blocks: int = 4
    kernel_size: int = 11
    dilation: int = 1


class SpliceAiConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`SpliceAiModel`][multimolecule.models.SpliceAiModel]. It is used to instantiate a SpliceAI model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the SpliceAI [Illumina/SpliceAI](https://github.com/Illumina/SpliceAI)
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the SpliceAI model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`SpliceAiModel`].
            Defaults to 5.
        context:
            The length of the context window. The input sequence will be padded with zeros of length `context // 2` on
            each side.
        hidden_size:
            Dimensionality of the encoder layers.
        stages:
            Configuration for each stage in the SpliceAI model. Each stage is a [`SpliceAiStageConfig`] object.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability for all convolution layers in the encoder.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of output labels.
        output_contexts:
            Whether to output the context vectors for each stage.

    Examples:
        >>> from multimolecule import SpliceAiConfig, SpliceAiModel
        >>> # Initializing a SpliceAI multimolecule/spliceai style configuration
        >>> configuration = SpliceAiConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/spliceai style configuration
        >>> model = SpliceAiModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "spliceai"

    def __init__(
        self,
        vocab_size: int = 4,
        context: int = 10000,
        hidden_size: int = 32,
        stages: list[SpliceAiStageConfig] | None = None,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.1,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        num_labels: int = 3,
        output_contexts: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context = context
        if stages is None:
            stages = [
                SpliceAiStageConfig(num_blocks=4, kernel_size=11),
                SpliceAiStageConfig(num_blocks=4, kernel_size=11, dilation=4),
                SpliceAiStageConfig(num_blocks=4, kernel_size=21, dilation=10),
                SpliceAiStageConfig(num_blocks=4, kernel_size=41, dilation=25),
            ]
        self.stages = stages
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.num_labels = num_labels
        self.output_contexts = output_contexts
