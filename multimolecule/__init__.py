# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from . import models, tokenisers
from .downstream.crispr_off_target import (
    AutoModelForCrisprOffTarget,
    RnaBertForCrisprOffTarget,
    RnaFmForCrisprOffTarget,
    RnaMsmForCrisprOffTarget,
    SpliceBertForCrisprOffTarget,
    UtrBertForCrisprOffTarget,
    UtrLmForCrisprOffTarget,
)
from .models import (
    AutoModelForNucleotideClassification,
    BaseHeadConfig,
    CaLmConfig,
    CaLmForMaskedLM,
    CaLmForNucleotideClassification,
    CaLmForPreTraining,
    CaLmForSequenceClassification,
    CaLmForTokenClassification,
    CaLmModel,
    HeadConfig,
    MaskedLMHeadConfig,
    PreTrainedConfig,
    RiNALMoConfig,
    RiNALMoForMaskedLM,
    RiNALMoForNucleotideClassification,
    RiNALMoForPreTraining,
    RiNALMoForSequenceClassification,
    RiNALMoForTokenClassification,
    RiNALMoModel,
    RnaBertConfig,
    RnaBertForMaskedLM,
    RnaBertForNucleotideClassification,
    RnaBertForPreTraining,
    RnaBertForSequenceClassification,
    RnaBertForTokenClassification,
    RnaBertModel,
    RnaFmConfig,
    RnaFmForMaskedLM,
    RnaFmForNucleotideClassification,
    RnaFmForPreTraining,
    RnaFmForSequenceClassification,
    RnaFmForTokenClassification,
    RnaFmModel,
    RnaMsmConfig,
    RnaMsmForMaskedLM,
    RnaMsmForNucleotideClassification,
    RnaMsmForPreTraining,
    RnaMsmForSequenceClassification,
    RnaMsmForTokenClassification,
    RnaMsmModel,
    SpliceBertConfig,
    SpliceBertForMaskedLM,
    SpliceBertForNucleotideClassification,
    SpliceBertForPreTraining,
    SpliceBertForSequenceClassification,
    SpliceBertForTokenClassification,
    SpliceBertModel,
    UtrBertConfig,
    UtrBertForMaskedLM,
    UtrBertForNucleotideClassification,
    UtrBertForPreTraining,
    UtrBertForSequenceClassification,
    UtrBertForTokenClassification,
    UtrBertModel,
    UtrLmConfig,
    UtrLmForMaskedLM,
    UtrLmForNucleotideClassification,
    UtrLmForPreTraining,
    UtrLmForSequenceClassification,
    UtrLmForTokenClassification,
    UtrLmModel,
)
from .module import (
    ClassificationHead,
    ContactPredictionHead,
    Criterion,
    HeadRegistry,
    HeadTransformRegistry,
    HeadTransformRegistryHF,
    IdentityTransform,
    LinearTransform,
    MaskedLMHead,
    NonLinearTransform,
    NucleotideClassificationHead,
    NucleotideHeadRegistryHF,
    NucleotideKMerHead,
    PositionEmbeddingRegistry,
    PositionEmbeddingRegistryHF,
    RotaryEmbedding,
    SequenceClassificationHead,
    SinusoidalEmbedding,
    TokenClassificationHead,
    TokenHeadRegistryHF,
    TokenKMerHead,
)
from .tokenisers import DnaTokenizer, ProteinTokenizer, RnaTokenizer
from .utils import count_parameters

__all__ = [
    "PreTrainedConfig",
    "HeadConfig",
    "BaseHeadConfig",
    "MaskedLMHeadConfig",
    "tokenisers",
    "DnaTokenizer",
    "RnaTokenizer",
    "ProteinTokenizer",
    "models",
    "AutoModelForNucleotideClassification",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "CaLmForSequenceClassification",
    "CaLmForTokenClassification",
    "CaLmForNucleotideClassification",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RiNALMoForSequenceClassification",
    "RiNALMoForTokenClassification",
    "RiNALMoForNucleotideClassification",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaBertForSequenceClassification",
    "RnaBertForTokenClassification",
    "RnaBertForNucleotideClassification",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaFmForSequenceClassification",
    "RnaFmForTokenClassification",
    "RnaFmForNucleotideClassification",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "RnaMsmForSequenceClassification",
    "RnaMsmForTokenClassification",
    "RnaMsmForNucleotideClassification",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "SpliceBertForSequenceClassification",
    "SpliceBertForTokenClassification",
    "SpliceBertForNucleotideClassification",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrBertForSequenceClassification",
    "UtrBertForTokenClassification",
    "UtrBertForNucleotideClassification",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "UtrLmForSequenceClassification",
    "UtrLmForTokenClassification",
    "UtrLmForNucleotideClassification",
    "AutoModelForCrisprOffTarget",
    "RnaBertForCrisprOffTarget",
    "RnaFmForCrisprOffTarget",
    "RnaMsmForCrisprOffTarget",
    "SpliceBertForCrisprOffTarget",
    "UtrBertForCrisprOffTarget",
    "UtrLmForCrisprOffTarget",
    "HeadRegistry",
    "ClassificationHead",
    "SequenceClassificationHead",
    "TokenHeadRegistryHF",
    "TokenClassificationHead",
    "TokenKMerHead",
    "NucleotideHeadRegistryHF",
    "NucleotideClassificationHead",
    "NucleotideKMerHead",
    "ContactPredictionHead",
    "MaskedLMHead",
    "HeadTransformRegistry",
    "HeadTransformRegistryHF",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
    "PositionEmbeddingRegistry",
    "PositionEmbeddingRegistryHF",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "Criterion",
    "count_parameters",
]
