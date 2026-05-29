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

from ...modules import HeadConfig
from ..configuration_utils import PreTrainedConfig


class DeltaSpliceLayerConfig(FlatDict):
    r"""
    Configuration for a single DeltaSplice dilated residual layer.

    Args:
        kernel_size:
            Convolution kernel size used by both convolutions in the residual layer.
        dilation:
            Dilation (atrous rate) used by both convolutions in the residual layer.
    """

    kernel_size: int = 11
    dilation: int = 1


class DeltaSpliceConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DeltaSpliceModel`][multimolecule.models.DeltaSpliceModel]. It is used to instantiate a DeltaSplice model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration similar to the official
    [chaolinzhanglab/DeltaSplice](https://github.com/chaolinzhanglab/DeltaSplice) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the DeltaSplice one-hot input channels. Defaults to 4 (`A`, `C`, `G`, `U`); the `N`
            padding token is encoded as all-zero channels.
        context:
            Number of flanking nucleotides represented around the requested output positions. The model pads
            `context // 2` zero-context positions on each side, reproducing the upstream fixed-window interface while
            returning one output per input token.
        hidden_size:
            Dimensionality of the convolutional encoder.
        layers:
            Configuration for each dilated residual layer. Each layer is a [`DeltaSpliceLayerConfig`] object.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and prediction heads.
        dropout:
            Dropout probability used between the two convolutions of each residual layer.
        batch_norm_eps:
            The epsilon used by batch normalization layers.
        batch_norm_momentum:
            The momentum used by batch normalization layers.
        num_ensemble:
            Number of internal checkpoint members averaged by the model. The official DeltaSplice releases provide
            five seed checkpoints per variant.
        num_labels:
            Number of splice-site usage labels (`no_splice`, `acceptor`, `donor`). Must be 3 for the official
            checkpoints.
        head:
            Configuration of the optional token prediction head.
        problem_type:
            Problem type for the optional token prediction head.
        output_contexts:
            Whether to output intermediate encoder representations.

    Examples:
        >>> from multimolecule.models.deltasplice import DeltaSpliceConfig, DeltaSpliceLayerConfig, DeltaSpliceModel
        >>> layer = DeltaSpliceLayerConfig(kernel_size=3, dilation=1)
        >>> configuration = DeltaSpliceConfig(context=4, hidden_size=8, layers=[layer], num_ensemble=1)
        >>> model = DeltaSpliceModel(configuration)
        >>> configuration = model.config
    """

    model_type = "deltasplice"

    pad_token_id: int = 4
    bos_token_id: int | None = None  # type: ignore[assignment]
    eos_token_id: int | None = None  # type: ignore[assignment]
    unk_token_id: int = 4
    mask_token_id: int | None = None  # type: ignore[assignment]
    null_token_id: int | None = None  # type: ignore[assignment]

    def __init__(
        self,
        vocab_size: int = 4,
        context: int = 30000,
        hidden_size: int = 64,
        layers: list[DeltaSpliceLayerConfig] | None = None,
        hidden_act: str = "relu",
        dropout: float = 0.3,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        num_ensemble: int = 5,
        num_labels: int = 3,
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
        if layers is None:
            kernels = [
                11,
                11,
                11,
                11,
                19,
                19,
                19,
                19,
                25,
                25,
                25,
                25,
                33,
                33,
                33,
                33,
                43,
                43,
                85,
                85,
                85,
                85,
                85,
                85,
            ]
            dilations = [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                8,
                8,
                8,
                8,
                16,
                16,
                16,
                16,
                16,
                16,
                32,
                32,
            ]
            layers = [
                DeltaSpliceLayerConfig(kernel_size=kernel_size, dilation=dilation)
                for kernel_size, dilation in zip(kernels, dilations)
            ]
        self.layers = [
            layer if isinstance(layer, DeltaSpliceLayerConfig) else DeltaSpliceLayerConfig(**layer) for layer in layers
        ]
        if num_labels != 3:
            raise ValueError(f"DeltaSplice emits three usage channels; `num_labels` must be 3, got {num_labels}.")
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
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.num_ensemble = num_ensemble
        self.problem_type = problem_type
        if head is None:
            head = HeadConfig(num_labels=num_labels, hidden_size=hidden_size, problem_type=problem_type)
        elif not isinstance(head, HeadConfig):
            head = HeadConfig(**head)
        self.head = head
        self.output_contexts = output_contexts

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if pad_token_id is not None and pad_token_id < vocab_size:
            raise ValueError(
                f"DeltaSplice expects pad_token_id ({pad_token_id}) outside the {vocab_size} nucleotide channels."
            )
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if context <= 0 or context % 2:
            raise ValueError(f"context must be a positive even integer, got {context}.")
        if num_ensemble <= 0:
            raise ValueError(f"num_ensemble must be positive, got {num_ensemble}.")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        for index, layer in enumerate(self.layers):
            if min(layer.kernel_size, layer.dilation) <= 0:
                raise ValueError(f"Layer {index} has non-positive kernel size or dilation: {layer}.")
            if layer.kernel_size % 2 == 0:
                raise ValueError(f"Layer {index} uses an even kernel size ({layer.kernel_size}); expected odd.")
        if self.context < self.convolution_reduction:
            raise ValueError(
                f"context ({self.context}) must be at least the encoder reduction ({self.convolution_reduction})."
            )
        if (self.context - self.convolution_reduction) % 2:
            raise ValueError(
                f"context ({self.context}) and encoder reduction ({self.convolution_reduction}) must have same parity."
            )

    @property
    def convolution_reduction(self) -> int:
        r"""Number of positions removed by the unpadded dilated convolutions in the encoder."""
        return 2 * sum((layer.kernel_size - 1) * layer.dilation for layer in self.layers)

    @property
    def encoder_crop(self) -> int:
        r"""Additional positions cropped on each side after the convolutional encoder."""
        return (self.context - self.convolution_reduction) // 2
