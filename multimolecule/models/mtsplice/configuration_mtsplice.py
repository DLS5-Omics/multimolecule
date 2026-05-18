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

from ...modules import HeadConfig
from ..configuration_utils import PreTrainedConfig


class MtSpliceConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`MtSpliceModel`][multimolecule.models.MtSpliceModel]. It is used to instantiate a MTSplice model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the MTSplice
    [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    MTSplice (Cheng et al. 2021) is the tissue-specific second generation of MMSplice. It scores a cassette exon
    together with its flanking introns through two parallel dilated-convolution towers: an *acceptor* (3' splice
    site) tower over the upstream region and a *donor* (5' splice site) tower over the downstream region. The two
    towers are positionally re-weighted by B-spline transformations, pooled, and combined by a small dense head into
    a tissue-resolved delta-logit-PSI splicing-effect score across 56 GTEx tissues.

    Args:
        vocab_size:
            Vocabulary size of the MTSplice model. Defines the number of feature channels derived from the one-hot
            encoded `input_ids`. Defaults to 4 (the `ACGT` nucleobase alphabet).
        hidden_size:
            Number of convolution filters in the two sequence towers.
        kernel_size:
            Kernel size of the first (stem) convolution in each tower.
        num_blocks:
            Number of residual dilated-convolution blocks per tower.
        block_kernel_size:
            Kernel size of the residual dilated-convolution blocks.
        dilation_base:
            Base of the exponentially growing dilation rate; block `i` uses dilation `dilation_base ** (i + 1)`.
        acceptor_length:
            Length (in bp) of the acceptor (3' splice site) input region, intron overhang plus exon flank.
        donor_length:
            Length (in bp) of the donor (5' splice site) input region, exon flank plus intron overhang.
        spline_bases:
            Number of B-spline bases used by the positional re-weighting layers.
        spline_degree:
            Polynomial degree of the B-spline bases.
        mlp_size:
            Hidden size of the dense head that maps pooled features to tissue scores.
        hidden_act:
            The non-linear activation function in the convolution towers and the dense head.
        batch_norm_eps:
            The epsilon used by the batch normalization layers. Defaults to 0.001 to match the upstream
            Keras `BatchNormalization` default.
        hidden_dropout:
            The dropout probability applied before the tissue projection.
        num_labels:
            Number of tissue outputs. MTSplice predicts delta-logit-PSI for the 56 GTEx tissues, so this defaults
            to 56.

    Examples:
        >>> from multimolecule import MtSpliceConfig, MtSpliceModel
        >>> # Initializing a MTSplice multimolecule/mtsplice style configuration
        >>> configuration = MtSpliceConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/mtsplice style configuration
        >>> model = MtSpliceModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "mtsplice"

    def __init__(
        self,
        vocab_size: int = 4,
        hidden_size: int = 64,
        kernel_size: int = 11,
        num_blocks: int = 8,
        block_kernel_size: int = 3,
        dilation_base: int = 2,
        acceptor_length: int = 400,
        donor_length: int = 400,
        spline_bases: int = 10,
        spline_degree: int = 3,
        mlp_size: int = 32,
        hidden_act: str = "relu",
        batch_norm_eps: float = 1e-3,
        hidden_dropout: float = 0.5,
        num_labels: int = 56,
        head: HeadConfig | None = None,
        problem_type: str | None = "regression",
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        if pad_token_id != vocab_size:
            raise ValueError(
                f"MTSplice expects `pad_token_id` ({pad_token_id}) to equal `vocab_size` ({vocab_size}) so "
                "`N` padding is encoded as all-zero input channels."
            )
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, **kwargs)
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.block_kernel_size = block_kernel_size
        self.dilation_base = dilation_base
        self.acceptor_length = acceptor_length
        self.donor_length = donor_length
        self.spline_bases = spline_bases
        self.spline_degree = spline_degree
        self.mlp_size = mlp_size
        self.hidden_act = hidden_act
        self.batch_norm_eps = batch_norm_eps
        self.hidden_dropout = hidden_dropout
        self.problem_type = problem_type
        if head is None:
            head = HeadConfig(num_labels=num_labels, hidden_size=num_labels, problem_type=problem_type)
        elif not isinstance(head, HeadConfig):
            head = HeadConfig(**head)
        self.head = head
