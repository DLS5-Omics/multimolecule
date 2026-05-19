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


class ChromBpNetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`ChromBpNetModel`][multimolecule.models.ChromBpNetModel]. It is used to instantiate a ChromBPNet model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the ChromBPNet
    [HEK293T-GFP](https://zenodo.org/records/16295014) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    ChromBPNet predicts base-resolution chromatin accessibility (ATAC-seq / DNase-seq) with explicit enzyme-bias
    correction. It internally composes two BPNet-style dilated-convolution sub-models:

    - a *bias* sub-model that captures the Tn5/DNase enzyme cleavage bias on chromatin background;
    - an *accessibility* sub-model that learns the bias-corrected accessibility signal.

    The final prediction is a single base-resolution task whose output is factorized into two terminal branches that
    share their respective backbones:

    - a *profile* branch producing per-position multinomial logits of shape
      `(batch_size, profile_length, num_tasks * num_strands)`;
    - a *count* branch producing a scalar per task and strand of shape `(batch_size, num_tasks * num_strands)`.

    The bias and accessibility sub-models are composed *internally*: their profile logits are added, and their count
    logits are combined in log/exp space (`logsumexp`). They are not a user-facing split.

    Args:
        vocab_size:
            Vocabulary size of the ChromBPNet model. Defines the number of one-hot input channels derived from
            `input_ids`. Defaults to 5 to match the MultiMolecule `streamline` DNA alphabet (`ACGTN`).
        sequence_length:
            The canonical input DNA sequence length in base pairs.
            Defaults to 2114.
        profile_length:
            The centered output profile length in base pairs.
            Defaults to 1000.
        hidden_size:
            Number of channels in the convolutional backbone of the accessibility sub-model.
        bias_hidden_size:
            Number of channels in the convolutional backbone of the bias sub-model.
        stem_kernel_size:
            Kernel size of the first (motif) convolution.
        num_dilated_layers:
            Number of dilated residual convolution blocks following the stem in the accessibility sub-model.
        bias_num_dilated_layers:
            Number of dilated residual convolution blocks following the stem in the bias sub-model.
        dilated_kernel_size:
            Kernel size of each dilated residual convolution.
        profile_kernel_size:
            Kernel size of the wide convolution in the profile branch.
        num_tasks:
            Number of prediction tasks.
        num_strands:
            Number of strands predicted per task. ChromBPNet ATAC/DNase predicts a single (unstranded) track.
        hidden_act:
            The non-linear activation function (function or string) in the backbones.
        count_loss_weight:
            The weight applied to the count regression loss when combining it with the profile multinomial loss.
        head:
            The configuration of the generic token prediction head. If not provided, it defaults to regression.
        output_hidden_states:
            Whether to output the backbone hidden states.

    Examples:
        >>> from multimolecule import ChromBpNetConfig, ChromBpNetModel
        >>> # Initializing a ChromBPNet multimolecule/chrombpnet style configuration
        >>> configuration = ChromBpNetConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/chrombpnet style configuration
        >>> model = ChromBpNetModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "chrombpnet"

    def __init__(
        self,
        vocab_size: int = 5,
        sequence_length: int = 2114,
        profile_length: int = 1000,
        hidden_size: int = 512,
        bias_hidden_size: int = 128,
        stem_kernel_size: int = 21,
        num_dilated_layers: int = 8,
        bias_num_dilated_layers: int = 4,
        dilated_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        num_tasks: int = 1,
        num_strands: int = 1,
        hidden_act: str = "relu",
        count_loss_weight: float = 1.0,
        head: HeadConfig | None = None,
        output_hidden_states: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        if num_dilated_layers < 1:
            raise ValueError(f"num_dilated_layers ({num_dilated_layers}) must be at least 1.")
        if bias_num_dilated_layers < 1:
            raise ValueError(f"bias_num_dilated_layers ({bias_num_dilated_layers}) must be at least 1.")
        required_profile_features = profile_length + profile_kernel_size - 1
        for name, layers in (
            ("num_dilated_layers", num_dilated_layers),
            ("bias_num_dilated_layers", bias_num_dilated_layers),
        ):
            receptive_field = (stem_kernel_size - 1) + sum(
                (dilated_kernel_size - 1) * 2 ** (i + 1) for i in range(layers)
            )
            valid_feature_length = sequence_length - receptive_field
            if valid_feature_length < required_profile_features:
                raise ValueError(
                    f"{name} leaves {valid_feature_length} valid positions, but the profile head needs at least "
                    f"{required_profile_features}."
                )
        if sequence_length < required_profile_features:
            raise ValueError(
                "sequence_length must be at least profile_length + profile_kernel_size - 1 "
                f"({required_profile_features}), but got {sequence_length}."
            )
        if profile_length < 1:
            raise ValueError(f"profile_length ({profile_length}) must be at least 1.")
        if num_tasks < 1:
            raise ValueError(f"num_tasks ({num_tasks}) must be at least 1.")
        if num_strands < 1:
            raise ValueError(f"num_strands ({num_strands}) must be at least 1.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.profile_length = profile_length
        self.hidden_size = hidden_size
        self.bias_hidden_size = bias_hidden_size
        self.stem_kernel_size = stem_kernel_size
        self.num_dilated_layers = num_dilated_layers
        self.bias_num_dilated_layers = bias_num_dilated_layers
        self.dilated_kernel_size = dilated_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.num_tasks = num_tasks
        self.num_strands = num_strands
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
        # ``PretrainedConfig.__init__`` assigns ``num_labels``; ChromBPNet derives it from
        # ``num_tasks * num_strands`` so the assignment is intentionally ignored.
        pass
