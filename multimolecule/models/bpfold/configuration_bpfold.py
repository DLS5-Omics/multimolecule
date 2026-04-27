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


class BpfoldConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`BpfoldModel`][multimolecule.models.BpfoldModel]. It is used to instantiate a BPfold model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the BPfold [heqin-zhu/BPfold](https://github.com/heqin-zhu/BPfold) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Token vocabulary size of the BPfold model.
        hidden_size:
            Dimensionality of nucleotide token embeddings and transformer hidden states.
        num_hidden_layers:
            Number of base-pair attention transformer blocks.
        attention_head_size:
            Hidden size per attention head.
        intermediate_size:
            Dimensionality of the feed-forward layer inside each transformer block.
        hidden_dropout:
            Dropout probability in transformer blocks.
        positional_embedding:
            Positional bias type used by self-attention. The original checkpoint uses `"dyn"`.
        num_pairwise_convolutions:
            Number of convolutional layers applied to the pairwise energy map.
        pairwise_kernel_size:
            Kernel size for pairwise energy convolutions.
        use_squeeze_excitation:
            Whether to use squeeze-and-excitation blocks in pairwise convolutions.
        use_base_pair_energy:
            Whether to use base-pair motif energy maps.
        use_base_pair_probability:
            Whether to use an externally provided base-pair probability map.
        separate_outer_inner_energy:
            Whether motif energy is represented as separate outer and inner energy maps.
        motif_radius:
            Number of neighboring bases in base-pair motifs. The published BPfold model uses three.
        max_length:
            Training-time maximum sequence length used by the original checkpoints.
        threshold:
            Probability threshold for predicting base pairs during post-processing.
        use_postprocessing:
            Whether to run the constrained BPfold post-processing loop in `forward`.
        postprocess_iterations:
            Number of constrained post-processing iterations.
        postprocess_lr_min:
            Learning rate for the minimization step in post-processing.
        postprocess_lr_max:
            Learning rate for the Lagrangian multiplier maximization step in post-processing.
        postprocess_rho:
            L1 sparsity coefficient used by canonical post-processing.
        postprocess_nc_rho:
            L1 sparsity coefficient used by non-canonical post-processing.
        postprocess_with_l1:
            Whether to apply L1 shrinkage in post-processing.
        postprocess_s:
            Logit cutoff used by canonical post-processing.
        postprocess_nc_s:
            Logit cutoff used by non-canonical post-processing.
        pos_weight:
            Positive-class weight used by the original weighted binary cross-entropy training loss.
        num_members:
            Number of internal checkpoint members in the released BPfold predictor.

    Examples:
        >>> from multimolecule import BpfoldConfig, BpfoldModel
        >>> configuration = BpfoldConfig()
        >>> model = BpfoldModel(configuration)
        >>> configuration = model.config
    """

    model_type = "bpfold"

    def __init__(
        self,
        vocab_size: int = 11,
        hidden_size: int = 256,
        num_hidden_layers: int = 12,
        attention_head_size: int = 32,
        intermediate_size: int = 768,
        hidden_dropout: float = 0.1,
        positional_embedding: str = "dyn",
        num_pairwise_convolutions: int = 3,
        pairwise_kernel_size: int = 3,
        use_squeeze_excitation: bool = True,
        use_base_pair_energy: bool = True,
        use_base_pair_probability: bool = False,
        separate_outer_inner_energy: bool = True,
        motif_radius: int = 3,
        max_length: int = 600,
        threshold: float = 0.5,
        use_postprocessing: bool = False,
        postprocess_iterations: int = 100,
        postprocess_lr_min: float = 0.01,
        postprocess_lr_max: float = 0.1,
        postprocess_rho: float = 1.6,
        postprocess_nc_rho: float = 0.5,
        postprocess_with_l1: bool = True,
        postprocess_s: float = 1.5,
        postprocess_nc_s: float = 0.5,
        pos_weight: float = 300.0,
        num_members: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_size % attention_head_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by attention_head_size ({attention_head_size})."
            )
        if positional_embedding not in {"dyn", "alibi"}:
            raise ValueError(f"positional_embedding must be 'dyn' or 'alibi', but got {positional_embedding!r}.")
        if motif_radius != 3:
            raise ValueError("BPfold currently supports the published 3-neighbor motif energy table only.")
        if num_members <= 0:
            raise ValueError(f"num_members must be positive, but got {num_members}.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_size = attention_head_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.positional_embedding = positional_embedding
        self.num_pairwise_convolutions = num_pairwise_convolutions
        self.pairwise_kernel_size = pairwise_kernel_size
        self.use_squeeze_excitation = use_squeeze_excitation
        self.use_base_pair_energy = use_base_pair_energy
        self.use_base_pair_probability = use_base_pair_probability
        self.separate_outer_inner_energy = separate_outer_inner_energy
        self.motif_radius = motif_radius
        self.max_length = max_length
        self.threshold = threshold
        self.use_postprocessing = use_postprocessing
        self.postprocess_iterations = postprocess_iterations
        self.postprocess_lr_min = postprocess_lr_min
        self.postprocess_lr_max = postprocess_lr_max
        self.postprocess_rho = postprocess_rho
        self.postprocess_nc_rho = postprocess_nc_rho
        self.postprocess_with_l1 = postprocess_with_l1
        self.postprocess_s = postprocess_s
        self.postprocess_nc_s = postprocess_nc_s
        self.pos_weight = pos_weight
        self.num_members = num_members
