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


class PangolinStageConfig(FlatDict):
    r"""
    Configuration for a single Pangolin stage.

    A stage is a contiguous group of dilated residual blocks that share a kernel size and dilation, followed by a
    skip-connection convolution.

    Args:
        num_blocks:
            Number of dilated residual blocks in the stage.
        kernel_size:
            Convolution kernel size for the blocks in the stage.
        dilation:
            Dilation (atrous rate) for the blocks in the stage.
    """

    num_blocks: int = 4
    kernel_size: int = 11
    dilation: int = 1


class PangolinConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`PangolinModel`][multimolecule.models.PangolinModel]. It is used to instantiate a Pangolin model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the Pangolin [tkzeng/Pangolin](https://github.com/tkzeng/Pangolin) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the Pangolin model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`PangolinModel`].
            Defaults to 5 (`A`, `C`, `G`, `U`, `N`).
        context:
            The length of the context window. The input sequence is padded with zeros of length `context // 2` on each
            side, and the encoder trims the same amount before the prediction head.
        hidden_size:
            Dimensionality of the encoder layers.
        stages:
            Configuration for each stage in the Pangolin model. Each stage is a [`PangolinStageConfig`] object.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_ensemble:
            Number of replicate networks averaged inside each tissue-specific model group. The official Pangolin v2
            release uses three replicates per tissue.
        num_tissues:
            Number of tissue-specific model groups. The official release predicts four tissues (heart, liver, brain,
            testis), each with a splice-site score (2 channels) and a splice-site usage score (1 channel), for a total
            of `num_tissues * 3` upstream output channels.
        num_labels:
            Number of output labels for the [`TokenPredictionHead`]. Defaults to 4, one per-base splice-site usage
            value per tissue.
        head:
            Configuration for the [`TokenPredictionHead`].
        problem_type:
            Problem type for the token prediction head.
        output_contexts:
            Whether to output the context vectors for each stage.

    Examples:
        >>> from multimolecule import PangolinConfig, PangolinModel
        >>> # Initializing a Pangolin multimolecule/pangolin style configuration
        >>> configuration = PangolinConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/pangolin style configuration
        >>> model = PangolinModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "pangolin"

    # Pangolin consumes raw nucleotide sequences (`A`, `C`, `G`, `U`, `N`) with no special tokens; `N` doubles
    # as the padding token. There is no beginning/end-of-sequence token to strip in the prediction head.
    pad_token_id: int = 4
    bos_token_id: int | None = None  # type: ignore[assignment]
    eos_token_id: int | None = None  # type: ignore[assignment]
    unk_token_id: int = 4
    mask_token_id: int | None = None  # type: ignore[assignment]
    null_token_id: int | None = None  # type: ignore[assignment]

    def __init__(
        self,
        vocab_size: int = 5,
        context: int = 10000,
        hidden_size: int = 32,
        stages: list[PangolinStageConfig] | None = None,
        hidden_act: str = "relu",
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        num_ensemble: int = 3,
        num_tissues: int = 4,
        num_labels: int = 4,
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
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, unk_token_id=unk_token_id, **kwargs)
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        self.mask_token_id = mask_token_id  # type: ignore[assignment]
        self.null_token_id = null_token_id  # type: ignore[assignment]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context = context
        if stages is None:
            stages = [
                PangolinStageConfig(num_blocks=4, kernel_size=11, dilation=1),
                PangolinStageConfig(num_blocks=4, kernel_size=11, dilation=4),
                PangolinStageConfig(num_blocks=4, kernel_size=21, dilation=10),
                PangolinStageConfig(num_blocks=4, kernel_size=41, dilation=25),
            ]
        self.stages = [
            stage if isinstance(stage, PangolinStageConfig) else PangolinStageConfig(**stage) for stage in stages
        ]
        self.hidden_act = hidden_act
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.num_ensemble = num_ensemble
        self.num_tissues = num_tissues
        self.problem_type = problem_type
        if head is None:
            head = HeadConfig(num_labels=num_labels, hidden_size=hidden_size, problem_type=problem_type)
        elif not isinstance(head, HeadConfig):
            head = HeadConfig(**head)
        self.head = head
        self.output_contexts = output_contexts

        if vocab_size <= pad_token_id:
            raise ValueError(f"vocab_size ({vocab_size}) must include pad_token_id ({pad_token_id}).")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if context <= 0 or context % 2:
            raise ValueError(f"context must be a positive even integer, got {context}.")
        if min(num_ensemble, num_tissues, num_labels) <= 0:
            raise ValueError("num_ensemble, num_tissues, and num_labels must be positive.")
        for index, stage in enumerate(self.stages):
            if min(stage.num_blocks, stage.kernel_size, stage.dilation) <= 0:
                raise ValueError(f"Stage {index} has non-positive block, kernel, or dilation values: {stage}.")
