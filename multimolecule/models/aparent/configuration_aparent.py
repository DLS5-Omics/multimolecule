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


class AparentConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`AparentModel`][multimolecule.models.AparentModel]. It is used to instantiate an APARENT model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the APARENT [johli/aparent](https://github.com/johli/aparent) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the APARENT model. Defines the number of input channels of the first convolution.
            Defaults to 5 (`A`, `C`, `G`, `U`, `N`), matching the MultiMolecule RNA `streamline` alphabet. The
            upstream checkpoint only uses the four canonical DNA channels, with upstream `T` exposed as `U`.
        sequence_length:
            The fixed 3'UTR/polyA input sequence length APARENT was trained on (205 nt).
        conv1_filters:
            Number of filters in the first convolution.
        conv1_kernel_size:
            Kernel size (sequence span) of the first convolution. The first convolution also spans the full
            nucleotide dimension.
        pool_size:
            Pooling window of the max-pooling layer after the first convolution.
        conv2_filters:
            Number of filters in the second convolution.
        conv2_kernel_size:
            Kernel size of the second convolution.
        hidden_sizes:
            Sizes of the two fully connected layers after the convolutional stack. The second value is the size of
            the shared sequence representation exposed as `pooler_output`.
        dropouts:
            Dropout probabilities applied after each fully connected layer.
        hidden_act:
            The non-linear activation function used by the convolutional and dense layers.
        num_isoform_labels:
            Dimension of the upstream isoform-proportion output (sigmoid). APARENT predicts a single scalar.
        num_cleavage_labels:
            Dimension of the upstream positional cleavage-distribution output (softmax). APARENT predicts 206
            positions (205 sequence positions + 1 distal/library bias slot).
        library_size:
            Size of the upstream one-hot library-identity input concatenated before the output layers. The
            MultiMolecule API keeps this as a non-persistent zero feature, matching the upstream default encoder.
        head:
            The configuration of the sequence-level prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching APARENT's APA isoform prediction task.

    Examples:
        >>> from multimolecule import AparentConfig, AparentModel
        >>> # Initializing a APARENT johli/aparent style configuration
        >>> configuration = AparentConfig()
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = AparentModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "aparent"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 205,
        conv1_filters: int = 96,
        conv1_kernel_size: int = 8,
        pool_size: int = 2,
        conv2_filters: int = 128,
        conv2_kernel_size: int = 6,
        hidden_sizes: list[int] | None = None,
        dropouts: list[float] | None = None,
        hidden_act: str = "relu",
        num_isoform_labels: int = 1,
        num_cleavage_labels: int = 206,
        library_size: int = 13,
        num_labels: int = 1,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        if dropouts is None:
            dropouts = [0.1, 0.1]
        if len(hidden_sizes) != 2 or len(dropouts) != 2:
            raise ValueError(
                f"APARENT expects exactly two dense layers; got hidden_sizes={hidden_sizes}, dropouts={dropouts}."
            )
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.conv1_filters = conv1_filters
        self.conv1_kernel_size = conv1_kernel_size
        self.pool_size = pool_size
        self.conv2_filters = conv2_filters
        self.conv2_kernel_size = conv2_kernel_size
        self.hidden_sizes = hidden_sizes
        self.dropouts = dropouts
        self.hidden_act = hidden_act
        self.num_isoform_labels = num_isoform_labels
        self.num_cleavage_labels = num_cleavage_labels
        self.library_size = library_size
        # ``hidden_size`` is the dimensionality of the shared dense representation
        # consumed by the MultiMolecule sequence-prediction head.
        self.hidden_size = hidden_sizes[-1]
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
