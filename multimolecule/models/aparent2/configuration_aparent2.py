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


class Aparent2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`Aparent2Model`][multimolecule.models.Aparent2Model]. It is used to instantiate a APARENT2 model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the APARENT2 [johli/aparent-resnet](https://github.com/johli/aparent-resnet)
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    APARENT2 is a residual convolutional network that predicts human 3' UTR Alternative Polyadenylation (APA) and
    cleavage magnitude at nucleotide resolution. The network is fully convolutional plus a position-wise
    locally-connected library-bias layer; it does not contain any flatten/dense layers.

    Args:
        vocab_size:
            Vocabulary size of the APARENT2 model. Defines the number of one-hot input channels derived from
            `input_ids`. Defaults to 5 (the MultiMolecule streamline RNA alphabet `ACGUN`). The converted first
            projection represents `N` as the upstream 0.25 mixture across A/C/G/U, with upstream T exposed as U.
        sequence_length:
            The fixed length of the polyadenylation signal sequence the model was trained on. APARENT2 expects a 205 nt
            window with the core hexamer (e.g. `AAUAAA`) starting at position 70 (0-indexed).
        hidden_size:
            Number of feature channels used throughout the residual network.
        num_groups:
            Number of residual-block groups.
        num_blocks:
            Number of residual blocks per group.
        kernel_size:
            Convolution kernel size used inside each residual block.
        dilations:
            Dilation factor for each residual-block group. Must have `num_groups` entries.
        num_libraries:
            Dimensionality of the one-hot training sub-library bias input.
        library_index:
            The training sub-library index used to construct the deterministic library-bias input. The upstream
            variant-effect workflow always uses index 11.
        hidden_act:
            The non-linear activation function used inside the residual blocks.
        batch_norm_eps:
            The epsilon used by the batch normalization layers.
        batch_norm_momentum:
            The momentum used by the batch normalization layers.
        num_labels:
            Number of output labels. APARENT2 predicts a cleavage distribution over `sequence_length + 1` positions
            (the extra position is the "no cleavage in window" bucket), so this defaults to 206.
        head:
            The configuration of the prediction head. Defaults to a regression head
            (`problem_type="regression"`), matching APARENT2's cleavage-distribution prediction task.

    Examples:
        >>> from multimolecule import Aparent2Config, Aparent2Model
        >>> # Initializing a APARENT2 multimolecule/aparent2 style configuration
        >>> configuration = Aparent2Config()
        >>> # Initializing a model (with random weights) from the multimolecule/aparent2 style configuration
        >>> model = Aparent2Model(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "aparent2"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 205,
        hidden_size: int = 32,
        num_groups: int = 7,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        num_libraries: int = 13,
        library_index: int = 11,
        hidden_act: str = "relu",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        num_labels: int = 206,
        head: HeadConfig | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        if dilations is None:
            dilations = [1, 2, 4, 8, 4, 2, 1]
        if len(dilations) != num_groups:
            raise ValueError(f"`dilations` must have `num_groups` ({num_groups}) entries, but got {len(dilations)}.")
        if not 0 <= library_index < num_libraries:
            raise ValueError(f"`library_index` ({library_index}) must be in [0, num_libraries={num_libraries}).")
        if num_labels != sequence_length + 1:
            raise ValueError(
                f"`num_labels` ({num_labels}) must equal `sequence_length + 1` ({sequence_length + 1}); "
                "APARENT2 predicts a cleavage distribution over `sequence_length + 1` positions."
            )
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.num_libraries = num_libraries
        self.library_index = library_index
        self.hidden_act = hidden_act
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
