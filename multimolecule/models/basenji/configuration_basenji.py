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


class BasenjiBlockConfig(FlatDict):
    r"""
    Configuration for the dilated residual tower of the Basenji2 trunk.

    Basenji2 stacks `num_blocks` dilated residual units. Each unit runs on a `hidden_size`-channel
    residual stream and internally bottlenecks to `bottleneck_size` channels for the dilated
    convolution before projecting back. The dilation factor starts at `dilation` and is multiplied
    by `dilation_rate` after every block (rounded to the nearest integer when `round_dilation` is
    set), which is how Basenji2 reaches the receptive field needed for ~131 kb input windows.

    Args:
        num_blocks:
            Number of dilated residual blocks in the tower.
        kernel_size:
            Kernel size of the dilated (bottleneck) convolution.
        bottleneck_size:
            Channel count of the dilated convolution bottleneck.
        dilation:
            Dilation factor of the first block.
        dilation_rate:
            Multiplicative factor applied to the dilation after each block.
        round_dilation:
            Whether to round the running dilation to the nearest integer after each multiply
            (upstream Basenji2 uses `round=true`).
        dropout:
            Dropout probability applied to the projected (return) convolution of every block.
    """

    num_blocks: int = 11
    kernel_size: int = 3
    bottleneck_size: int = 384
    dilation: int = 1
    dilation_rate: float = 1.5
    round_dilation: bool = True
    dropout: float = 0.3


class BasenjiConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`BasenjiModel`][multimolecule.models.BasenjiModel]. It is used to instantiate a Basenji model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    configuration that faithfully reproduces the upstream Basenji2 human graph
    ([calico/basenji](https://github.com/calico/basenji), `manuscripts/cross2020/params_human.json`).

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Basenji2 predicts genomic coverage tracks at a *binned* resolution. A long DNA window of
    `sequence_length` base pairs is downsampled by the convolution + pooling stem and tower, then
    cropped by `crop_bins` bins on each side, so the output has shape
    `(batch_size, num_bins, num_labels)` where `num_labels` is the number of coverage tracks.

    Args:
        vocab_size:
            Vocabulary size of the Basenji model. Defines the number of input feature channels
            derived from the MultiMolecule DNA token order.
            Defaults to 5 (`A`, `C`, `G`, `T`, `N`).
        sequence_length:
            The length, in base pairs, of the input DNA window.
            Defaults to 131072 (~131 kb).
        stem_channels:
            Number of channels produced by the first (stem) convolution.
        stem_kernel_size:
            Kernel size of the first (stem) convolution.
        stem_pool_size:
            Pooling size applied after every convolution block in the stem and tower.
        conv_tower_channels:
            Explicit per-stage output channel schedule of the reducing convolution tower. Basenji2
            grows the width as ``339, 399, 470, 554, 652, 768``; the tower length is
            ``len(conv_tower_channels)`` and each stage halves the resolution.
        conv_tower_kernel_size:
            Kernel size used by every convolution in the reducing tower.
        blocks:
            Configuration of the dilated residual tower. A single [`BasenjiBlockConfig`].
        crop_bins:
            Number of bins trimmed from *each* side of the binned axis after the dilated tower
            (upstream `Cropping1D`).
        head_hidden_size:
            Channel count of the final pointwise convolution block feeding the track head.
        hidden_act:
            The non-linear activation used throughout the network. Basenji2 uses the
            tanh-approximation GELU (`gelu_new`).
        head_act:
            The activation applied to the final track projection. Basenji2 uses `softplus`.
        hidden_dropout:
            Dropout probability of the final pointwise convolution block.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers (PyTorch convention; upstream Keras
            momentum 0.9 corresponds to PyTorch momentum 0.1).
        num_labels:
            Number of genomic coverage tracks predicted per bin.
            Defaults to 5313 (the human track set released with Basenji2).
        head:
            The configuration of the binned track prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching Basenji's genomic coverage prediction task.
        output_contexts:
            Whether to output the context vectors for each tower block.

    Examples:
        >>> from multimolecule import BasenjiConfig, BasenjiModel
        >>> # Initializing a Basenji multimolecule/basenji style configuration
        >>> configuration = BasenjiConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/basenji style configuration
        >>> model = BasenjiModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "basenji"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 131072,
        stem_channels: int = 288,
        stem_kernel_size: int = 15,
        stem_pool_size: int = 2,
        conv_tower_channels: list[int] | None = None,
        conv_tower_kernel_size: int = 5,
        blocks: BasenjiBlockConfig | None = None,
        crop_bins: int = 64,
        head_hidden_size: int = 1536,
        hidden_act: str = "gelu_new",
        head_act: str = "softplus",
        hidden_dropout: float = 0.05,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 5313,
        head: HeadConfig | None = None,
        output_contexts: bool = False,
        **kwargs,
    ):
        # Basenji is a feature-channel DNA model: it consumes a raw one-hot DNA window with no
        # special tokens, and its output is binned coverage tracks. There is no BOS/EOS/MASK token
        # on either the input or the binned positional axis, so the shared TokenPredictionHead must
        # not trim "special" bins.
        kwargs.setdefault("bos_token_id", None)
        kwargs.setdefault("eos_token_id", None)
        kwargs.setdefault("mask_token_id", None)
        kwargs.setdefault("null_token_id", None)
        super().__init__(num_labels=num_labels, **kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.stem_channels = stem_channels
        self.stem_kernel_size = stem_kernel_size
        self.stem_pool_size = stem_pool_size
        if conv_tower_channels is None:
            conv_tower_channels = [339, 399, 470, 554, 652, 768]
        self.conv_tower_channels = list(conv_tower_channels)
        self.conv_tower_kernel_size = conv_tower_kernel_size
        if blocks is None:
            blocks = BasenjiBlockConfig()
        self.blocks = blocks if isinstance(blocks, BasenjiBlockConfig) else BasenjiBlockConfig(**dict(blocks))
        self.crop_bins = crop_bins
        self.head_hidden_size = head_hidden_size
        self.hidden_act = hidden_act
        self.head_act = head_act
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
        self.output_contexts = output_contexts

        if self.stem_pool_size < 1:
            raise ValueError(f"stem_pool_size must be >= 1, got {self.stem_pool_size}")
        if self.stem_channels < 1:
            raise ValueError(f"stem_channels must be >= 1, got {self.stem_channels}")
        if any(c < 1 for c in self.conv_tower_channels):
            raise ValueError(f"conv_tower_channels must be positive, got {self.conv_tower_channels}")
        if self.blocks.bottleneck_size < 1:
            raise ValueError(f"blocks.bottleneck_size must be >= 1, got {self.blocks.bottleneck_size}")
        if self.crop_bins < 0:
            raise ValueError(f"crop_bins must be >= 0, got {self.crop_bins}")
        if self.pool_factor <= 0:
            raise ValueError(f"pool_factor must be positive, got {self.pool_factor}")
        if self.sequence_length % self.pool_factor != 0:
            raise ValueError(
                f"sequence_length ({self.sequence_length}) must be divisible by the total pooling "
                f"factor ({self.pool_factor}) so the binned output is well defined."
            )
        if self.num_bins <= 0:
            raise ValueError(
                f"crop_bins ({self.crop_bins}) trims the entire binned axis "
                f"(pre-crop bins = {self.sequence_length // self.pool_factor}); reduce crop_bins."
            )

    @property
    def num_pool_layers(self) -> int:
        r"""Number of pooling stages: the stem block plus every reducing-tower stage."""
        return 1 + len(self.conv_tower_channels)

    @property
    def pool_factor(self) -> int:
        r"""Total downsampling factor applied by the stem and tower, i.e. base pairs per bin."""
        return self.stem_pool_size**self.num_pool_layers

    @property
    def hidden_size(self) -> int:
        r"""Channel count of the dilated residual stream."""
        return self.conv_tower_channels[-1] if self.conv_tower_channels else self.stem_channels

    @property
    def num_bins(self) -> int:
        r"""Number of output bins along the positional (token) axis, after cropping."""
        return self.sequence_length // self.pool_factor - 2 * self.crop_bins
