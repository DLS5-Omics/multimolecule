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

from ..configuration_utils import PreTrainedConfig


class UfoldConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`UfoldModel`][multimolecule.models.UfoldModel]. It is used to instantiate a UFold model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the UFold [uci-cbcl/UFold](https://github.com/uci-cbcl/UFold) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Token vocabulary size of the UFold model. Defaults to 5 for the `A/C/G/U/N` tokenizer vocabulary.
        input_channels:
            Number of image-like input channels. The original UFold model uses 16 outer-product base-pair channels
            plus one hand-crafted pairing-score channel.
        output_channels:
            Number of U-Net output channels. The original UFold model predicts one contact-score matrix.
        channel_sizes:
            U-Net channel sizes for the five original resolution levels.
        min_size:
            Minimum padded image size used before the U-Net. The original short-sequence dataset pads to 80.
        size_multiple:
            Spatial size multiple required by the four downsampling stages.
        batch_norm_eps:
            Epsilon used by the BatchNorm2d layers.
        batch_norm_momentum:
            Momentum used by the BatchNorm2d layers.
        threshold:
            Probability threshold for predicting base pairs during post-processing.
        use_postprocessing:
            Whether to run the UFold constrained post-processing loop in `forward`.
        postprocess_iterations:
            Number of constrained post-processing iterations.
        postprocess_lr_min:
            Learning rate for the minimization step in UFold post-processing.
        postprocess_lr_max:
            Learning rate for the Lagrangian multiplier maximization step in UFold post-processing.
        postprocess_rho:
            L1 sparsity coefficient used by UFold post-processing.
        postprocess_with_l1:
            Whether to apply L1 shrinkage in UFold post-processing.
        postprocess_s:
            Logit cutoff used by UFold post-processing. Defaults to `log(9)`, the original value.
        allow_noncanonical:
            Whether post-processing should allow non-canonical base pairs.
        pos_weight:
            Positive-class weight used by the original weighted binary cross-entropy training loss.

    Examples:
        >>> from multimolecule import UfoldConfig, UfoldModel
        >>> configuration = UfoldConfig()
        >>> model = UfoldModel(configuration)
        >>> configuration = model.config
    """

    model_type = "ufold"

    def __init__(
        self,
        vocab_size: int = 5,
        input_channels: int = 17,
        output_channels: int = 1,
        channel_sizes: list[int] | None = None,
        min_size: int = 80,
        size_multiple: int = 16,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        threshold: float = 0.5,
        use_postprocessing: bool = False,
        postprocess_iterations: int = 100,
        postprocess_lr_min: float = 0.01,
        postprocess_lr_max: float = 0.1,
        postprocess_rho: float = 1.6,
        postprocess_with_l1: bool = True,
        postprocess_s: float = 2.1972245773362196,
        allow_noncanonical: bool = False,
        pos_weight: float = 300.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if input_channels != 17:
            raise ValueError(f"UFold expects 17 input channels, but got {input_channels}.")
        if output_channels != 1:
            raise ValueError(f"UFold expects one output channel, but got {output_channels}.")
        if channel_sizes is None:
            channel_sizes = [32, 64, 128, 256, 512]
        if len(channel_sizes) != 5:
            raise ValueError(f"UFold expects five channel sizes, but got {len(channel_sizes)}.")
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, but got {min_size}.")
        if size_multiple <= 0:
            raise ValueError(f"size_multiple must be positive, but got {size_multiple}.")

        self.vocab_size = vocab_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channel_sizes = channel_sizes
        self.min_size = min_size
        self.size_multiple = size_multiple
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.threshold = threshold
        self.use_postprocessing = use_postprocessing
        self.postprocess_iterations = postprocess_iterations
        self.postprocess_lr_min = postprocess_lr_min
        self.postprocess_lr_max = postprocess_lr_max
        self.postprocess_rho = postprocess_rho
        self.postprocess_with_l1 = postprocess_with_l1
        self.postprocess_s = postprocess_s
        self.allow_noncanonical = allow_noncanonical
        self.pos_weight = pos_weight
