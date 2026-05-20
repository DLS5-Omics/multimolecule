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


class FramepoolConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`FramepoolModel`][multimolecule.models.FramepoolModel]. It is used to instantiate a Framepool model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the Framepool ``combined_residual`` architecture released with the
    [Karollus et al., 2021](https://doi.org/10.1371/journal.pcbi.1008982) paper.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Number of one-hot input channels derived from the MultiMolecule RNA tokenizer. Defaults to 5
            (``A``, ``C``, ``G``, ``U``, ``N``), matching the MultiMolecule RNA ``streamline`` alphabet. The upstream
            checkpoint only learns the four canonical channels, with upstream ``T`` exposed as ``U``; the embedding
            zeroes the ``N`` channel, matching the upstream ``compute_pad_mask`` semantics.
        null_channel_id:
            Channel index that represents the upstream "no nucleobase" token (``N`` and padding). The embedding
            zeroes this column so that the upstream ``pad_mask = sum(one_hot, axis=2)`` mask correctly identifies
            padded positions. Set to ``None`` to keep all channels.
        num_conv_layers:
            Number of stacked length-preserving residual convolutions in the encoder.
        conv_channels:
            Number of output channels for every convolution in the encoder.
        kernel_size:
            Kernel sizes of the encoder convolutions. Either a scalar shared across all layers, or a list with one
            entry per layer.
        dilations:
            Dilation rates of the encoder convolutions. Either a scalar shared across all layers, or a list with one
            entry per layer.
        hidden_act:
            Non-linear activation applied after each encoder convolution.
        padding:
            Convolution padding mode. ``same`` keeps the sequence length; ``causal`` left-pads to retain it.
        skip_connections:
            ``residual`` adds the input of every conv past the first to its output (the configuration used by the
            released checkpoint). ``""`` disables skip connections.
        num_dense_layers:
            Number of fully-connected layers between the frame-pooled representation and the unscaled MRL output.
        dense_sizes:
            Hidden sizes of the fully-connected layers. Length must match ``num_dense_layers``.
        dense_dropout:
            Dropout probability applied after every fully-connected layer.
        only_max_pool:
            If ``True``, the frame pooler concatenates only the per-frame global max pooled features (3 vectors).
            Otherwise it additionally concatenates the masked global average pooled features (6 vectors), matching
            the released checkpoint.
        library_size:
            Number of training sub-libraries supported by the scaling regression head. The released checkpoint was
            trained jointly on the ``egfp_unmod_1`` and ``random`` libraries, so ``library_size = 2``.
        library_index:
            Default training sub-library index used to construct the one-hot library indicator at inference. Matches
            the ``random`` library used for variant effect prediction upstream (Kipoi ``UTRVariantEffectModel``).
        num_labels:
            Number of scalar outputs of the model. Framepool predicts a single scalar mean ribosome load value.
        head:
            Configuration of the [`FramepoolForSequencePrediction`] regression head.

    Examples:
        >>> from multimolecule import FramepoolConfig, FramepoolModel
        >>> # Initializing a Framepool combined_residual style configuration
        >>> configuration = FramepoolConfig()
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = FramepoolModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "framepool"

    def __init__(
        self,
        vocab_size: int = 5,
        null_channel_id: int | None = 4,
        num_conv_layers: int = 3,
        conv_channels: int = 128,
        kernel_size: int | list[int] = 7,
        dilations: int | list[int] = 1,
        hidden_act: str = "relu",
        padding: str = "same",
        skip_connections: str = "residual",
        num_dense_layers: int = 1,
        dense_sizes: list[int] | None = None,
        dense_dropout: float = 0.2,
        only_max_pool: bool = False,
        library_size: int = 2,
        library_index: int = 1,
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if num_labels != 1:
            raise ValueError(f"Framepool predicts one mean-ribosome-load scalar, got num_labels={num_labels}.")
        if num_conv_layers <= 0:
            raise ValueError(f"num_conv_layers must be positive, got {num_conv_layers}.")
        kernel_size_list = [kernel_size] * num_conv_layers if isinstance(kernel_size, int) else list(kernel_size)
        dilations_list = [dilations] * num_conv_layers if isinstance(dilations, int) else list(dilations)
        if len(kernel_size_list) != num_conv_layers:
            raise ValueError(
                f"kernel_size must have num_conv_layers={num_conv_layers} entries, got {len(kernel_size_list)}."
            )
        if len(dilations_list) != num_conv_layers:
            raise ValueError(
                f"dilations must have num_conv_layers={num_conv_layers} entries, got {len(dilations_list)}."
            )
        if dense_sizes is None:
            dense_sizes = [64] * num_dense_layers
        if len(dense_sizes) != num_dense_layers:
            raise ValueError(
                f"dense_sizes must have num_dense_layers={num_dense_layers} entries, got {len(dense_sizes)}."
            )
        if padding not in ("same", "causal"):
            raise ValueError(f"padding must be 'same' or 'causal', got {padding!r}.")
        if skip_connections not in ("", "residual"):
            raise ValueError(f"skip_connections must be '' or 'residual', got {skip_connections!r}.")
        if library_size <= 0:
            raise ValueError(f"library_size must be positive, got {library_size}.")
        if not 0 <= library_index < library_size:
            raise ValueError(
                f"library_index must satisfy 0 <= library_index < library_size={library_size}, got {library_index}."
            )

        if null_channel_id is not None and not 0 <= null_channel_id < vocab_size:
            raise ValueError(
                f"null_channel_id must satisfy 0 <= null_channel_id < vocab_size={vocab_size}, "
                f"got {null_channel_id}."
            )

        self.vocab_size = vocab_size
        self.null_channel_id = null_channel_id
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size_list
        self.dilations = dilations_list
        self.hidden_act = hidden_act
        self.padding = padding
        self.skip_connections = skip_connections
        self.num_dense_layers = num_dense_layers
        self.dense_sizes = dense_sizes
        self.dense_dropout = dense_dropout
        self.only_max_pool = only_max_pool
        self.library_size = library_size
        self.library_index = library_index
        if head is None:
            head = HeadConfig(num_labels=num_labels, problem_type="regression")
        elif not isinstance(head, HeadConfig):
            head = HeadConfig(**head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the frame-pooled representation consumed by the prediction head.

        Derived from ``conv_channels`` and ``only_max_pool``; read-only to prevent silent drift.
        ``only_max_pool=True`` concatenates 3 per-frame max-pooled vectors; otherwise 6 (max + mean).
        """
        num_pools = 3 if self.only_max_pool else 6
        return self.conv_channels * num_pools
