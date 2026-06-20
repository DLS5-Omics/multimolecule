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

SUPPORTED_MOLECULES = ("dna", "rna")
SUPPORTED_POOLINGS = ("max", "maxavg")


class DeepBindConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DeepBindModel`][multimolecule.models.DeepBindModel]. It is used to instantiate a DeepBind model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the DeepBind [jisraeli/DeepBind](https://github.com/jisraeli/DeepBind) architecture
    distributed through Kipoi.

    DeepBind is a single-layer convolutional model that scores how strongly a DNA- or RNA-binding protein binds to a
    given sequence. The 538 trained TF/RBP models published with the original tool all share the same architecture; the
    variation lies in the training data (one protein per model) and the per-model filter / hidden width. Pick the per-
    protein checkpoint from the Hub (for example `multimolecule/deepbind-ctcf`) and load it through `from_pretrained`.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the DeepBind model. DeepBind consumes a one-hot encoding of the four nucleotides, so
            this also defines the number of input channels of the convolution.
            Defaults to 4.
        molecule:
            Which nucleic-acid alphabet the per-protein checkpoint scores: `"dna"` for transcription factors,
            `"rna"` for RNA-binding proteins. Determines which MultiMolecule tokenizer the converter saves with the
            checkpoint; the channel layout is always 4-wide.
        num_filters:
            Number of motif detectors (Conv1D filters).
        kernel_size:
            Motif width (Conv1D kernel size) used by the motif detectors.
        pooling:
            Sequence-pooling mode after the convolutional motif scan. `"max"` keeps only the global max; `"maxavg"`
            concatenates global max and global average and is the default mode in the released DeepBind tool.
        num_hidden:
            Width of the optional intermediate fully-connected layer. Set to 0 for the published "no hidden unit"
            mode, in which case the pooled feature vector is projected directly to the binding score. The default
            of 32 matches the most common DeepBind configuration.
        hidden_act:
            The non-linear activation function (function or string) applied after the convolutional and hidden
            fully-connected layers. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability for the hidden fully-connected layer.
        num_labels:
            Number of output labels. DeepBind predicts a single binding score per sequence, so this must be 1.
        head:
            The configuration of the prediction head. Defaults to a regression head (`problem_type="regression"`).

    Examples:
        >>> from multimolecule import DeepBindConfig, DeepBindModel
        >>> # Initializing a DeepBind multimolecule/deepbind-ctcf style configuration
        >>> configuration = DeepBindConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/deepbind-ctcf style configuration
        >>> model = DeepBindModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "deepbind"

    def __init__(
        self,
        vocab_size: int = 4,
        molecule: str = "dna",
        num_filters: int = 16,
        kernel_size: int = 24,
        pooling: str = "maxavg",
        num_hidden: int = 32,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if molecule not in SUPPORTED_MOLECULES:
            raise ValueError(f"`molecule` must be one of {SUPPORTED_MOLECULES}, got {molecule!r}.")
        if pooling not in SUPPORTED_POOLINGS:
            raise ValueError(f"`pooling` must be one of {SUPPORTED_POOLINGS}, got {pooling!r}.")
        if num_filters <= 0:
            raise ValueError(f"`num_filters` must be positive, got {num_filters}.")
        if kernel_size <= 0:
            raise ValueError(f"`kernel_size` must be positive, got {kernel_size}.")
        if num_hidden < 0:
            raise ValueError(f"`num_hidden` must be non-negative, got {num_hidden}.")
        if num_labels != 1:
            raise ValueError(f"DeepBind emits a single binding score; `num_labels` must be 1, got {num_labels}.")
        self.vocab_size = vocab_size
        self.molecule = molecule
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.num_hidden = num_hidden
        # `hidden_size` is the width of the feature vector the prediction head sees: the hidden FC
        # output when present, otherwise the pooled feature vector itself.
        self.hidden_size = num_hidden if num_hidden > 0 else self.pooled_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.num_labels = num_labels
        self.problem_type = "regression"
        if head is None:
            head = HeadConfig(num_labels=1, problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
            if head.num_labels is None:
                head.num_labels = 1
        self.head = head

    @property
    def pooled_size(self) -> int:
        r"""Width of the pooled feature vector produced by the convolutional motif scan."""
        return self.num_filters * (2 if self.pooling == "maxavg" else 1)
