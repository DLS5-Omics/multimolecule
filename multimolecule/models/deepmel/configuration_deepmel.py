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


class DeepMelConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DeepMelModel`][multimolecule.models.DeepMelModel]. It is used to instantiate a DeepMEL model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the DeepMEL [aertslab/DeepMEL](https://github.com/aertslab/DeepMEL) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the DeepMEL model. Defines the number of feature channels in the one-hot encoded input
            fed to the first convolution.
            Defaults to 5.
        input_length:
            The fixed length (in base pairs) of the input DNA sequence.
            Defaults to 500.
        conv_channels:
            Number of output channels (filters) of the first convolution.
            Defaults to 128.
        conv_kernel_size:
            Convolution kernel size.
            Defaults to 20.
        pool_size:
            Max-pool window applied after the convolution. The convolution stride is 1 and the pool stride matches the
            pool size, so the effective downsampling factor equals `pool_size`.
            Defaults to 10.
        time_distributed_channels:
            Hidden size of the time-distributed dense layer applied after pooling.
            Defaults to 128.
        lstm_hidden_size:
            Hidden size of each direction of the bidirectional LSTM.
            Defaults to 128.
        fc_dim:
            Hidden size of the fully-connected layer between the recurrent stack and the prediction head.
            Defaults to 256.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        conv_dropout:
            The dropout probability after the convolutional max-pool block.
        recurrent_dropout:
            The dropout probability after the bidirectional LSTM.
        fc_dropout:
            The dropout probability after the fully-connected layer.
        lstm_dropout:
            The dropout probability applied to the LSTM input weights during training.
        lstm_recurrent_dropout:
            The dropout probability applied to the LSTM recurrent weights during training.
        num_labels:
            Number of multi-label binary topics. DeepMEL predicts 24 melanoma topics (4 MEL + 7 MES + others).
            Defaults to 24.
        head:
            The configuration of the prediction head. Defaults to a multi-label binary classification head
            (`problem_type="multilabel"`), matching DeepMEL's chromatin-topic prediction task.

    Examples:
        >>> from multimolecule import DeepMelConfig, DeepMelModel
        >>> # Initializing a DeepMEL multimolecule/deepmel style configuration
        >>> configuration = DeepMelConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/deepmel style configuration
        >>> model = DeepMelModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "deepmel"

    def __init__(
        self,
        vocab_size: int = 5,
        input_length: int = 500,
        conv_channels: int = 128,
        conv_kernel_size: int = 20,
        pool_size: int = 10,
        time_distributed_channels: int = 128,
        lstm_hidden_size: int = 128,
        fc_dim: int = 256,
        hidden_act: str = "relu",
        conv_dropout: float = 0.2,
        recurrent_dropout: float = 0.2,
        fc_dropout: float = 0.4,
        lstm_dropout: float = 0.1,
        lstm_recurrent_dropout: float = 0.1,
        num_labels: int = 24,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}.")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}.")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}.")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels must be positive, got {conv_channels}.")
        if input_length < conv_kernel_size:
            raise ValueError(f"input_length ({input_length}) must be at least conv_kernel_size ({conv_kernel_size}).")
        if not 0.0 <= conv_dropout < 1.0:
            raise ValueError(f"conv_dropout must be in [0, 1), got {conv_dropout}.")
        if not 0.0 <= recurrent_dropout < 1.0:
            raise ValueError(f"recurrent_dropout must be in [0, 1), got {recurrent_dropout}.")
        if not 0.0 <= fc_dropout < 1.0:
            raise ValueError(f"fc_dropout must be in [0, 1), got {fc_dropout}.")
        if not 0.0 <= lstm_dropout < 1.0:
            raise ValueError(f"lstm_dropout must be in [0, 1), got {lstm_dropout}.")
        if not 0.0 <= lstm_recurrent_dropout < 1.0:
            raise ValueError(f"lstm_recurrent_dropout must be in [0, 1), got {lstm_recurrent_dropout}.")
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.time_distributed_channels = time_distributed_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_dim = fc_dim
        # The model's pooled representation (fed into the prediction head) is the per-branch fully-connected output,
        # averaged across the forward and reverse-complement branches. The decoder then maps this to the 24 topics.
        self.hidden_size = fc_dim
        self.hidden_act = hidden_act
        self.conv_dropout = conv_dropout
        self.recurrent_dropout = recurrent_dropout
        self.fc_dropout = fc_dropout
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        # DeepMEL performs multi-label binary classification of 24 chromatin topics. The MultiMolecule `problem_type`
        # convention lives on the head config, since the Transformers base config only accepts the HF `problem_type`
        # literals.
        if head is None:
            head = HeadConfig(problem_type="multilabel")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "multilabel"
        self.head = head

    @property
    def pooled_length(self) -> int:
        """Sequence length after the convolution (valid padding) and max-pooling step."""
        return (self.input_length - self.conv_kernel_size + 1) // self.pool_size

    @property
    def flattened_size(self) -> int:
        """Number of features produced by the per-branch flatten step (input width to the FC layer)."""
        return self.pooled_length * 2 * self.lstm_hidden_size
