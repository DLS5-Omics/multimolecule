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

from multimolecule.modules import HeadConfig

from ..configuration_utils import PreTrainedConfig


class BpNetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`BpNetModel`][multimolecule.models.BpNetModel]. It is used to instantiate a BPNet model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the BPNet [BPNet-OSKN](https://zenodo.org/records/4294904) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    BPNet predicts a single base-resolution signal task whose output is factorized into two terminal branches that
    share the dilated-convolution backbone:

    - a *profile* branch producing per-position multinomial logits of shape
      `(batch_size, sequence_length, num_tasks * num_strands)`;
    - a *count* branch producing a scalar per task and strand of shape `(batch_size, num_tasks * num_strands)`.

    Args:
        vocab_size:
            Vocabulary size of the BPNet model. Defines the number of one-hot input channels derived from `input_ids`.
            Defaults to 5 to match the MultiMolecule `streamline` DNA alphabet (`ACGTN`).
        hidden_size:
            Number of channels in the convolutional backbone.
        stem_kernel_size:
            Kernel size of the first (motif) convolution.
        num_dilated_layers:
            Number of dilated residual convolution blocks following the stem.
        dilated_kernel_size:
            Kernel size of each dilated residual convolution.
        profile_kernel_size:
            Kernel size of the transposed convolution in the profile branch.
        num_tasks:
            Number of prediction tasks (e.g. transcription factors).
        num_strands:
            Number of strands predicted per task.
        hidden_act:
            The non-linear activation function (function or string) in the backbone.
        count_loss_weight:
            The weight applied to the count regression loss when combining it with the profile multinomial loss.
        head:
            The configuration of the generic token prediction head. If not provided, it defaults to regression.
        output_hidden_states:
            Whether to output the backbone hidden states.

    Examples:
        >>> from multimolecule import BpNetConfig, BpNetModel
        >>> # Initializing a BPNet multimolecule/bpnet style configuration
        >>> configuration = BpNetConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/bpnet style configuration
        >>> model = BpNetModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "bpnet"

    def __init__(
        self,
        vocab_size: int = 5,
        hidden_size: int = 64,
        stem_kernel_size: int = 25,
        num_dilated_layers: int = 9,
        dilated_kernel_size: int = 3,
        profile_kernel_size: int = 25,
        num_tasks: int = 4,
        num_strands: int = 2,
        hidden_act: str = "relu",
        count_loss_weight: float = 1.0,
        head: HeadConfig | None = None,
        output_hidden_states: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        self.num_tasks = num_tasks
        self.num_strands = num_strands
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        if num_dilated_layers < 1:
            raise ValueError(f"num_dilated_layers ({num_dilated_layers}) must be at least 1.")
        if num_tasks < 1:
            raise ValueError(f"num_tasks ({num_tasks}) must be at least 1.")
        if num_strands < 1:
            raise ValueError(f"num_strands ({num_strands}) must be at least 1.")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.stem_kernel_size = stem_kernel_size
        self.num_dilated_layers = num_dilated_layers
        self.dilated_kernel_size = dilated_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.hidden_act = hidden_act
        self.count_loss_weight = count_loss_weight
        if head is None:
            head = HeadConfig(problem_type="regression")
        else:
            head = HeadConfig(head)
            if head.problem_type is None:
                head.problem_type = "regression"
        self.head = head
        self.output_hidden_states = output_hidden_states

    @property
    def num_labels(self) -> int:
        return self.num_tasks * self.num_strands

    @num_labels.setter
    def num_labels(self, value: int) -> None:
        # ``PretrainedConfig.__init__`` assigns ``num_labels``; BPNet derives it from
        # ``num_tasks * num_strands`` so the assignment is intentionally ignored.
        pass
