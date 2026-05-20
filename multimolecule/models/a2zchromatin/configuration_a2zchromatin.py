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


class A2zChromatinConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an
    [`A2zChromatinModel`][multimolecule.models.A2zChromatinModel]. It is used to instantiate an a2z-chromatin model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the a2z-chromatin
    [twrightsman/a2z-regulatory](https://github.com/twrightsman/a2z-regulatory) architecture (DanQ topology trained on
    angiosperm chromatin data, distributed via Kipoi as `a2z-accessibility` and `a2z-methylation`).

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the a2z-chromatin model. Upstream a2z-chromatin consumes four nucleotide channels, but
            the converted MultiMolecule checkpoint expands the first convolution to the DNA IUPAC tokenizer alphabet so
            ambiguity tokens reproduce upstream fractional one-hot encodings.
            Defaults to 16.
        sequence_length:
            The fixed length of the input DNA sequence in base pairs.
            Defaults to 600.
        conv_channels:
            Number of filters in the first (and only) 1D convolution.
            Defaults to 320.
        conv_kernel_size:
            Kernel size of the 1D convolution.
            Defaults to 26.
        conv_dropout:
            Dropout probability applied after the convolution.
            Defaults to 0.2.
        pool_size:
            Max-pool window size and stride applied after the convolution.
            Defaults to 13.
        lstm_hidden_size:
            Hidden dimensionality of each direction of the bidirectional LSTM.
            Defaults to 320.
        lstm_dropout:
            Dropout probability applied after the bidirectional LSTM.
            Defaults to 0.5.
        fc_size:
            Hidden dimensionality of the fully-connected layer between the LSTM and the prediction head.
            Defaults to 925.
        hidden_act:
            The non-linear activation function (function or string) applied after the convolution. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
            Defaults to `"relu"`.
        num_labels:
            Number of output labels. a2z-chromatin predicts a single binary target (chromatin accessibility for the
            `a2z-accessibility` variant, lack of DNA methylation for the `a2z-methylation` variant).
            Defaults to 1.
        head:
            The configuration of the prediction head. Defaults to a binary classification head
            (`problem_type="binary"`), matching a2z-chromatin's per-window accessibility / unmethylation task.

    Examples:
        >>> from multimolecule import A2zChromatinConfig, A2zChromatinModel
        >>> # Initializing an a2z-chromatin multimolecule/a2zchromatin-accessibility style configuration
        >>> configuration = A2zChromatinConfig()
        >>> # Initializing a random-weight model from the multimolecule/a2zchromatin-accessibility style configuration
        >>> model = A2zChromatinModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "a2zchromatin"

    def __init__(
        self,
        vocab_size: int = 16,
        sequence_length: int = 600,
        conv_channels: int = 320,
        conv_kernel_size: int = 26,
        conv_dropout: float = 0.2,
        pool_size: int = 13,
        lstm_hidden_size: int = 320,
        lstm_dropout: float = 0.5,
        fc_size: int = 925,
        hidden_act: str = "relu",
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, but got {sequence_length}.")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels must be positive, but got {conv_channels}.")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, but got {conv_kernel_size}.")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, but got {pool_size}.")
        if lstm_hidden_size <= 0:
            raise ValueError(f"lstm_hidden_size must be positive, but got {lstm_hidden_size}.")
        if fc_size <= 0:
            raise ValueError(f"fc_size must be positive, but got {fc_size}.")
        # Upstream DanQ topology uses valid (zero-padding) convolution followed by floor-mode pooling with
        # stride == pool_size; require the configured (sequence_length, conv_kernel_size, pool_size) triple to
        # leave at least one pooled position so the BLSTM has a non-empty input window.
        conv_out_length = sequence_length - conv_kernel_size + 1
        if conv_out_length <= 0:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be greater than conv_kernel_size ({conv_kernel_size})."
            )
        pooled_length = conv_out_length // pool_size
        if pooled_length <= 0:
            raise ValueError(
                f"The configured (sequence_length={sequence_length}, conv_kernel_size={conv_kernel_size}, "
                f"pool_size={pool_size}) leaves no positions after pooling."
            )
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_dropout = conv_dropout
        self.pool_size = pool_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_dropout = lstm_dropout
        self.fc_size = fc_size
        self.hidden_size = fc_size
        self.hidden_act = hidden_act
        # a2z-chromatin performs per-window binary prediction (accessibility or unmethylation). The MultiMolecule
        # `problem_type` convention lives on the head config, since the Transformers base config only accepts the
        # HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="binary")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "binary"
        self.head = head
