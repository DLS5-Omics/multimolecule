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


class FactorNetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`FactorNetModel`][multimolecule.models.FactorNetModel]. It is used to instantiate a FactorNet model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a configuration that reproduces the upstream FactorNet
    [uci-cbcl/FactorNet](https://github.com/uci-cbcl/FactorNet) architecture for the CTCF/HEK293 release (the
    `meta_RNAseq_Unique35_DGF` variant).

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    FactorNet is a Siamese-style model: the same convolution / BLSTM / dense stack is applied to the forward and the
    reverse-complement of the input window, the two scalar outputs are averaged, and the result is the per-TF binding
    probability. The MultiMolecule port shares this contract: a single `(batch_size, sequence_length)` `input_ids`
    tensor encodes the forward strand, the reverse-complement is computed inside the model, and the auxiliary
    per-position signals (e.g. DNase-seq, mappability) are passed via `auxiliary_signal`; per-cell-type metadata
    features (RNA-seq principal components and the like) are passed via `metadata_features`.

    Args:
        vocab_size:
            Vocabulary size of the FactorNet model. Defines the number of one-hot feature channels derived from the
            MultiMolecule DNA token order. FactorNet treats out-of-vocabulary nucleotides as all-zero rows.
            Defaults to 4 (`A`, `C`, `G`, `T`).
        sequence_length:
            The length, in base pairs, of the input DNA window consumed by the convolutional stack.
            Defaults to 1002 (matches the released CTCF/HEK293 release).
        num_auxiliary_signals:
            Number of auxiliary per-position signal channels concatenated with the one-hot DNA tensor before the first
            convolution. Upstream FactorNet pre-computed two per-position signals (Unique35 mappability and DGF DNase
            cleavage). Set to 0 to disable auxiliary signal injection (DNA-only).
            Defaults to 2.
        num_metadata_features:
            Number of per-window metadata features (RNA-seq principal components and similar) concatenated with the
            flattened convolutional representation before the second dense layer. Set to 0 to disable the metadata
            branch.
            Defaults to 8.
        conv_kernel_size:
            Kernel size (`filter_length`) of the first 1D convolution.
            Defaults to 34.
        conv_channels:
            Number of filters (`nb_filter`) of the first 1D convolution. Also the output dimensionality of the
            pointwise (`TimeDistributed(Dense)`) layer applied immediately after the convolution.
            Defaults to 128.
        conv_dropout:
            Dropout probability applied after the first convolution.
            Defaults to 0.1.
        pool_size:
            Stride and width of the max-pool layer that follows the pointwise dense layer.
            Defaults to 17.
        lstm_hidden_size:
            Number of hidden units of each direction of the bidirectional LSTM (the upstream `LSTM(output_dim=64)`).
            Set to 0 to omit the BLSTM layer.
            Defaults to 64.
        post_lstm_dropout:
            Dropout probability applied after the bidirectional LSTM (or after the post-pool dropout when the BLSTM is
            omitted).
            Defaults to 0.5.
        fc_hidden_size:
            Hidden size of the two fully-connected layers preceding the per-TF sigmoid head.
            Defaults to 128.
        hidden_act:
            The non-linear activation function used throughout the convolution and dense stack. If a string, `"gelu"`,
            `"relu"`, `"silu"`, and `"gelu_new"` are supported.
            Defaults to `"relu"`.
        num_labels:
            Number of per-window output labels. The released single-task FactorNet models predict one transcription
            factor (`num_labels=1`); the multi-task `multiTask_DGF` release predicts up to 19 TFs at once.
            Defaults to 1.
        head:
            The configuration of the prediction head. Defaults to a multi-label binary classification head
            (`problem_type="multilabel"`), matching FactorNet's per-TF binding probability output.

    Examples:
        >>> from multimolecule import FactorNetConfig, FactorNetModel
        >>> # Initializing a FactorNet multimolecule/factornet style configuration
        >>> configuration = FactorNetConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/factornet style configuration
        >>> model = FactorNetModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "factornet"

    def __init__(
        self,
        vocab_size: int = 4,
        sequence_length: int = 1002,
        num_auxiliary_signals: int = 2,
        num_metadata_features: int = 8,
        conv_kernel_size: int = 34,
        conv_channels: int = 128,
        conv_dropout: float = 0.1,
        pool_size: int = 17,
        lstm_hidden_size: int = 64,
        post_lstm_dropout: float = 0.5,
        fc_hidden_size: int = 128,
        hidden_act: str = "relu",
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        # FactorNet is a feature-channel DNA model: there are no BOS/EOS/MASK tokens and no learned word embeddings.
        # `N` (and other ambiguous IUPAC codes) maps to an all-zero one-hot row, which is also the natural padding
        # value, so there is no dedicated padding token id.
        kwargs.setdefault("bos_token_id", None)
        kwargs.setdefault("eos_token_id", None)
        kwargs.setdefault("mask_token_id", None)
        kwargs.setdefault("null_token_id", None)
        super().__init__(num_labels=num_labels, **kwargs)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, but got {sequence_length}.")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, but got {conv_kernel_size}.")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels must be positive, but got {conv_channels}.")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, but got {pool_size}.")
        if fc_hidden_size <= 0:
            raise ValueError(f"fc_hidden_size must be positive, but got {fc_hidden_size}.")
        if lstm_hidden_size < 0:
            raise ValueError(f"lstm_hidden_size must be non-negative, but got {lstm_hidden_size}.")
        if num_auxiliary_signals < 0:
            raise ValueError(f"num_auxiliary_signals must be non-negative, but got {num_auxiliary_signals}.")
        if num_metadata_features < 0:
            raise ValueError(f"num_metadata_features must be non-negative, but got {num_metadata_features}.")
        conv_output_length = sequence_length - conv_kernel_size + 1
        if conv_output_length < pool_size:
            raise ValueError(
                f"sequence_length ({sequence_length}) is too short for conv_kernel_size ({conv_kernel_size}) and "
                f"pool_size ({pool_size}); the convolution would not produce enough timesteps to pool."
            )
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_auxiliary_signals = num_auxiliary_signals
        self.num_metadata_features = num_metadata_features
        self.conv_kernel_size = conv_kernel_size
        self.conv_channels = conv_channels
        self.conv_dropout = conv_dropout
        self.pool_size = pool_size
        self.lstm_hidden_size = lstm_hidden_size
        self.post_lstm_dropout = post_lstm_dropout
        self.fc_hidden_size = fc_hidden_size
        self.hidden_act = hidden_act
        # `hidden_size` is the width of the pooled representation consumed by `SequencePredictionHead`; it is the
        # output dimensionality of the final fully-connected layer before the sigmoid head.
        self.hidden_size = fc_hidden_size
        # FactorNet predicts per-TF binding probabilities (multi-label binary classification). MultiMolecule's
        # `problem_type` convention lives on the head config, since the Transformers base config only accepts the
        # HF `problem_type` literals.
        if head is None:
            head = HeadConfig(problem_type="multilabel")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "multilabel"
        self.head = head

    @property
    def num_input_channels(self) -> int:
        r"""Total number of channels consumed by the first convolution: one-hot DNA + auxiliary per-position signals."""
        return self.vocab_size + self.num_auxiliary_signals

    @property
    def pooled_length(self) -> int:
        r"""Number of timesteps fed into the BLSTM / flatten, after the post-convolution max pool."""
        # FactorNet uses a Keras `valid` Conv1D followed by a `valid` MaxPool1D with stride == pool_length, matching
        # `length // pool_size` after floor division of the convolution's output length.
        return (self.sequence_length - self.conv_kernel_size + 1) // self.pool_size

    @property
    def flattened_size(self) -> int:
        r"""Width of the flattened tensor consumed by the first fully-connected layer."""
        per_step_channels = self.lstm_hidden_size * 2 if self.lstm_hidden_size > 0 else self.conv_channels
        return self.pooled_length * per_step_channels
