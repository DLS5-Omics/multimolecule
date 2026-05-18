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


class OpenSpliceAiStageConfig(FlatDict):
    r"""
    Configuration for a single OpenSpliceAI stage.

    Args:
        num_blocks:
            Number of residual convolutional blocks in the stage.
        kernel_size:
            Convolution kernel size for the stage.
        dilation:
            Dilation (atrous) factor for the stage.
    """

    num_blocks: int = 4
    kernel_size: int = 11
    dilation: int = 1


class OpenSpliceAiConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`OpenSpliceAiModel`][multimolecule.models.OpenSpliceAiModel]. It is used to instantiate an OpenSpliceAI model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OpenSpliceAI
    [Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI) `openspliceai-mane` 10000nt architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the OpenSpliceAI model. Defines the number of different tokens that can be represented
            by the `input_ids` passed when calling [`OpenSpliceAiModel`].
            Defaults to 4 (the one-hot nucleotide channels `A`, `C`, `G`, `T`).
        context:
            The length of the context window. The input sequence will be padded with zeros of length `context // 2` on
            each side so that the per-nucleotide output keeps the input resolution.
        hidden_size:
            Dimensionality of the encoder layers.
        stages:
            Configuration for each stage in the OpenSpliceAI model. Each stage is a [`OpenSpliceAiStageConfig`] object.
        hidden_act:
            The non-linear activation function (function or string) in the encoder. String values are resolved through
            `transformers.activations.ACT2FN`.
        hidden_act_kwargs:
            Keyword arguments used when instantiating string activations. Defaults to `{"negative_slope": 0.1}` for
            `"leaky_relu"` to match the original OpenSpliceAI checkpoints.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of output labels (neither / acceptor / donor).
        head:
            The configuration of the prediction head.
        output_contexts:
            Whether to output the context vectors for each stage.

    Examples:
        >>> from multimolecule import OpenSpliceAiConfig, OpenSpliceAiModel
        >>> # Initializing an OpenSpliceAI multimolecule/openspliceai style configuration
        >>> configuration = OpenSpliceAiConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/openspliceai style configuration
        >>> model = OpenSpliceAiModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "openspliceai"

    def __init__(
        self,
        vocab_size: int = 4,
        context: int = 10000,
        hidden_size: int = 32,
        stages: list[OpenSpliceAiStageConfig] | None = None,
        hidden_act: str = "leaky_relu",
        hidden_act_kwargs: dict[str, object] | None = None,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 3,
        head: HeadConfig | None = None,
        output_contexts: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        # OpenSpliceAI consumes raw one-hot nucleotide channels and does not use BOS/EOS tokens;
        # `pad_token_id` points at the `N` token (the last entry of the streamline DNA alphabet).
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        if hidden_act_kwargs is None and hidden_act == "leaky_relu":
            hidden_act_kwargs = {"negative_slope": 0.1}
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context = context
        if stages is None:
            stages = [
                OpenSpliceAiStageConfig(num_blocks=4, kernel_size=11, dilation=1),
                OpenSpliceAiStageConfig(num_blocks=4, kernel_size=11, dilation=4),
                OpenSpliceAiStageConfig(num_blocks=4, kernel_size=21, dilation=10),
                OpenSpliceAiStageConfig(num_blocks=4, kernel_size=41, dilation=25),
            ]
        self.stages = stages
        self.hidden_act = hidden_act
        self.hidden_act_kwargs = hidden_act_kwargs or {}
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        # OpenSpliceAI performs per-nucleotide multi-class classification (neither / acceptor / donor).
        self.head = HeadConfig(head) if head is not None else HeadConfig(problem_type="multiclass")
        self.output_contexts = output_contexts

    @property
    def cropping(self) -> int:
        r"""Total number of context nucleotides removed by the model (``2 * sum(dilation * (kernel_size - 1))``)."""
        return 2 * sum(s["dilation"] * (s["kernel_size"] - 1) * s["num_blocks"] for s in self.stages)
