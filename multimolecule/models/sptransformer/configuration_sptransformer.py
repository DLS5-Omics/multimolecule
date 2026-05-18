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

from ..configuration_utils import HeadConfig, PreTrainedConfig


class SpTransformerFeatureEncoderConfig(FlatDict):
    r"""
    Configuration for a single SpliceAI-style convolutional feature encoder used by SpTransformer.

    SpTransformer reuses two pre-trained dilated-residual convolutional encoders to extract per-position
    sequence features. Each encoder is a stack of dilated residual blocks; the feature map is taken before
    the encoder's own output projections.

    Args:
        hidden_size:
            Number of channels in the encoder.
    """

    hidden_size: int = 128


class SpTransformerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`SpTransformerModel`][multimolecule.models.SpTransformerModel]. It is used to instantiate a SpTransformer model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SpliceTransformer
    [ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the SpTransformer model. Defines the number of different tokens that can be represented
            by the `input_ids` passed when calling [`SpTransformerModel`].
            Defaults to 5 (`A`, `C`, `G`, `T`, `N`).
        context:
            The length of the context window. The encoder consumes `context` nucleotides of flanking context on each
            side of every predicted position.
        hidden_size:
            Dimensionality of the trainable input-projection path.
        encoders:
            Configuration for each SpliceAI-style convolutional feature encoder. Each encoder is a
            [`SpTransformerFeatureEncoderConfig`] object.
        attention_hidden_size:
            Dimensionality of the Sinkhorn transformer attention block.
        num_hidden_layers:
            Number of layers in the Sinkhorn transformer attention block.
        num_attention_heads:
            Number of attention heads in the Sinkhorn transformer attention block.
        num_local_attention_heads:
            Number of attention heads that use local (windowed) attention instead of Sinkhorn attention.
        intermediate_size:
            Dimensionality of the feed-forward layers in the attention block.
        bucket_size:
            Token bucket size for Sinkhorn / local attention.
        max_seq_len:
            Maximum sequence length consumed by the attention block. The concatenated features are
            center-cropped or padded to this length before the attention block.
        num_splice_labels:
            Number of splice-site score channels predicted by the original output head (no-splice, acceptor,
            donor).
        num_tissues:
            Number of tissues for which per-position splice-site usage is predicted by the original output head.
        hidden_act:
            The non-linear activation function (function or string) in the SpliceAI-style feature encoders.
        intermediate_act:
            The non-linear activation function (function or string) in the transformer feed-forward layers.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of output labels for the [`TokenPredictionHead`]. Defaults to 15, one per-position
            tissue-specific splice-site usage value.
        head:
            Configuration for the [`TokenPredictionHead`].
        problem_type:
            Problem type for the token prediction head.
        output_contexts:
            Whether to output the per-position attention-block representation.

    Examples:
        >>> from multimolecule import SpTransformerConfig, SpTransformerModel
        >>> # Initializing a SpTransformer multimolecule/sptransformer style configuration
        >>> configuration = SpTransformerConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/sptransformer style configuration
        >>> model = SpTransformerModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "sptransformer"

    # SpTransformer consumes raw nucleotide sequences (`A`, `C`, `G`, `T`, `N`) with no special tokens; `N`
    # doubles as the padding token. Converted checkpoints map the `N` input channel to zero weights.
    pad_token_id: int = 4
    bos_token_id: int | None = None  # type: ignore[assignment]
    eos_token_id: int | None = None  # type: ignore[assignment]
    unk_token_id: int = 4
    mask_token_id: int | None = None  # type: ignore[assignment]
    null_token_id: int | None = None  # type: ignore[assignment]

    def __init__(
        self,
        vocab_size: int = 5,
        context: int = 4000,
        hidden_size: int = 128,
        encoders: list[SpTransformerFeatureEncoderConfig] | None = None,
        attention_hidden_size: int = 256,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_local_attention_heads: int = 2,
        intermediate_size: int = 1024,
        bucket_size: int = 64,
        max_seq_len: int = 8192,
        num_splice_labels: int = 3,
        num_tissues: int = 15,
        hidden_act: str = "relu",
        intermediate_act: str = "gelu",
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 15,
        head: HeadConfig | None = None,
        problem_type: str | None = "regression",
        output_contexts: bool = False,
        pad_token_id: int = 4,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        unk_token_id: int = 4,
        mask_token_id: int | None = None,
        null_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        self.mask_token_id = mask_token_id  # type: ignore[assignment]
        self.null_token_id = null_token_id  # type: ignore[assignment]
        self.vocab_size = vocab_size
        self.context = context
        self.hidden_size = hidden_size
        if encoders is None:
            encoders = [
                SpTransformerFeatureEncoderConfig(hidden_size=128),
                SpTransformerFeatureEncoderConfig(hidden_size=64),
            ]
        self.encoders = [
            (
                encoder
                if isinstance(encoder, SpTransformerFeatureEncoderConfig)
                else SpTransformerFeatureEncoderConfig(**encoder)
            )
            for encoder in encoders
        ]
        self.attention_hidden_size = attention_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_local_attention_heads = num_local_attention_heads
        self.intermediate_size = intermediate_size
        self.bucket_size = bucket_size
        self.max_seq_len = max_seq_len
        self.num_splice_labels = num_splice_labels
        self.num_tissues = num_tissues
        self.hidden_act = hidden_act
        self.intermediate_act = intermediate_act
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.problem_type = problem_type
        if head is None:
            head = HeadConfig(num_labels=num_labels, hidden_size=attention_hidden_size, problem_type=problem_type)
        elif not isinstance(head, HeadConfig):
            head = HeadConfig(**head)
        self.head = head
        self.output_contexts = output_contexts

        if pad_token_id is not None and vocab_size <= pad_token_id:
            raise ValueError(f"vocab_size ({vocab_size}) must include pad_token_id ({pad_token_id}).")
        if context < 0:
            raise ValueError(f"context must be non-negative, got {context}.")
        min_dimension = min(
            hidden_size,
            attention_hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            bucket_size,
            max_seq_len,
            num_splice_labels,
            num_tissues,
            num_labels,
        )
        if min_dimension <= 0:
            raise ValueError(
                "hidden_size, attention_hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, "
                "bucket_size, max_seq_len, num_splice_labels, num_tissues, and num_labels must be positive."
            )
        if num_local_attention_heads < 0:
            raise ValueError(f"num_local_attention_heads must be non-negative, got {num_local_attention_heads}.")
        if attention_hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"attention_hidden_size ({attention_hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if num_local_attention_heads > num_attention_heads:
            raise ValueError(
                f"num_local_attention_heads ({num_local_attention_heads}) cannot exceed "
                f"num_attention_heads ({num_attention_heads})"
            )
        if max_seq_len % bucket_size != 0:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by bucket_size ({bucket_size})")
        for index, encoder in enumerate(self.encoders):
            if encoder.hidden_size <= 0:
                raise ValueError(f"Encoder {index} has non-positive hidden_size: {encoder.hidden_size}.")
