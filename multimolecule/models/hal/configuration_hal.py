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


class HalConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`HalModel`][multimolecule.models.HalModel]. It is used to instantiate a HAL model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the HAL model from
    [Learning the Sequence Determinants of Alternative Splicing from Millions of Random
    Sequences](https://doi.org/10.1016/j.cell.2015.09.054).

    HAL (Hexamer Additive Linear model) is a linear model over hexamer (k-mer) features that predicts alternative
    splicing outcomes such as 5' splice-site usage. The model weights are a published table of hexamer coefficients.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the HAL model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`HalModel`]. Only the four canonical nucleotides contribute hexamer
            features; remaining ids are ignored when counting hexamers.
            Defaults to 5.
        kmer_size:
            The k-mer (hexamer) size used for feature extraction. The published HAL model uses hexamers (`kmer_size=6`).
        nucleobase_size:
            Number of canonical nucleotides used to enumerate k-mers. The number of k-mer features is
            `nucleobase_size ** kmer_size`.
        region_length:
            The length of the sequence region scored by the model. The published HAL/Kipoi model scores a fixed
            160-nucleotide 5' splice-site window.
        hidden_size:
            Size of the scalar feature consumed by the optional sequence prediction loss wrapper. HAL emits one score,
            so this must be 1.
        num_labels:
            Number of output labels. HAL is a single-output regression model, so this defaults to 1.

    Examples:
        >>> from multimolecule import HalConfig, HalModel
        >>> # Initializing a HAL multimolecule/hal style configuration
        >>> configuration = HalConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/hal style configuration
        >>> model = HalModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "hal"

    def __init__(
        self,
        vocab_size: int = 5,
        kmer_size: int = 6,
        nucleobase_size: int = 4,
        region_length: int = 160,
        hidden_size: int = 1,
        head: HeadConfig | None = None,
        num_labels: int = 1,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, **kwargs)
        if vocab_size < 5:
            raise ValueError(f"vocab_size ({vocab_size}) must cover the streamline DNA alphabet `ACGTN`")
        if kmer_size != 6:
            raise ValueError(f"The published HAL checkpoint is a hexamer model; `kmer_size` must be 6, got {kmer_size}")
        if nucleobase_size != 4:
            raise ValueError(
                f"The published HAL checkpoint enumerates four canonical nucleotides; "
                f"`nucleobase_size` must be 4, got {nucleobase_size}"
            )
        if region_length != 160:
            raise ValueError(
                f"The published HAL checkpoint scores a fixed 160-nucleotide window; "
                f"`region_length` must be 160, got {region_length}"
            )
        if hidden_size != 1:
            raise ValueError(f"HAL emits a single scalar feature; `hidden_size` must be 1, got {hidden_size}")
        if num_labels != 1:
            raise ValueError(f"HAL emits a single score; `num_labels` must be 1, got {num_labels}")
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        self.vocab_size = vocab_size
        self.kmer_size = kmer_size
        self.nucleobase_size = nucleobase_size
        self.region_length = region_length
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.problem_type = "regression"
        self.head = HeadConfig(head) if head is not None else HeadConfig(num_labels=1, problem_type="regression")

    @property
    def num_kmers(self) -> int:
        r"""Number of distinct k-mer features (`nucleobase_size ** kmer_size`)."""
        return self.nucleobase_size**self.kmer_size

    @property
    def num_regions(self) -> int:
        r"""Number of position-specific HAL coefficient regions in the published artifact."""
        return 8

    @property
    def num_features(self) -> int:
        r"""Number of normalized k-mer frequency features consumed by the HAL linear layer."""
        return self.num_kmers

    @property
    def feature_size(self) -> int:
        r"""Alias for `num_features`, matching the feature vector consumed by the HAL linear layer."""
        return self.num_features
