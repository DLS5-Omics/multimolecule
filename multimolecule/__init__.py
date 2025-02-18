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


from .data import Dataset
from .models import (
    AutoModelForContactPrediction,
    AutoModelForNucleotidePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
    CaLmConfig,
    CaLmForContactPrediction,
    CaLmForMaskedLM,
    CaLmForNucleotidePrediction,
    CaLmForPreTraining,
    CaLmForSequencePrediction,
    CaLmForTokenPrediction,
    CaLmModel,
    ErnieRnaConfig,
    ErnieRnaForContactPrediction,
    ErnieRnaForMaskedLM,
    ErnieRnaForNucleotidePrediction,
    ErnieRnaForPreTraining,
    ErnieRnaForSecondaryStructurePrediction,
    ErnieRnaForSequencePrediction,
    ErnieRnaForTokenPrediction,
    ErnieRnaModel,
    PreTrainedConfig,
    RiNALMoConfig,
    RiNALMoForContactPrediction,
    RiNALMoForMaskedLM,
    RiNALMoForNucleotidePrediction,
    RiNALMoForPreTraining,
    RiNALMoForSequencePrediction,
    RiNALMoForTokenPrediction,
    RiNALMoModel,
    RnaBertConfig,
    RnaBertForContactPrediction,
    RnaBertForMaskedLM,
    RnaBertForNucleotidePrediction,
    RnaBertForPreTraining,
    RnaBertForSequencePrediction,
    RnaBertForTokenPrediction,
    RnaBertModel,
    RnaErnieConfig,
    RnaErnieForContactPrediction,
    RnaErnieForMaskedLM,
    RnaErnieForNucleotidePrediction,
    RnaErnieForPreTraining,
    RnaErnieForSequencePrediction,
    RnaErnieForTokenPrediction,
    RnaErnieModel,
    RnaFmConfig,
    RnaFmForContactPrediction,
    RnaFmForMaskedLM,
    RnaFmForNucleotidePrediction,
    RnaFmForPreTraining,
    RnaFmForSequencePrediction,
    RnaFmForTokenPrediction,
    RnaFmModel,
    RnaMsmConfig,
    RnaMsmForContactPrediction,
    RnaMsmForMaskedLM,
    RnaMsmForNucleotidePrediction,
    RnaMsmForPreTraining,
    RnaMsmForSequencePrediction,
    RnaMsmForTokenPrediction,
    RnaMsmModel,
    SpliceBertConfig,
    SpliceBertForContactPrediction,
    SpliceBertForMaskedLM,
    SpliceBertForNucleotidePrediction,
    SpliceBertForPreTraining,
    SpliceBertForSequencePrediction,
    SpliceBertForTokenPrediction,
    SpliceBertModel,
    UtrBertConfig,
    UtrBertForContactPrediction,
    UtrBertForMaskedLM,
    UtrBertForNucleotidePrediction,
    UtrBertForPreTraining,
    UtrBertForSequencePrediction,
    UtrBertForTokenPrediction,
    UtrBertModel,
    UtrLmConfig,
    UtrLmForContactPrediction,
    UtrLmForMaskedLM,
    UtrLmForNucleotidePrediction,
    UtrLmForPreTraining,
    UtrLmForSequencePrediction,
    UtrLmForTokenPrediction,
    UtrLmModel,
    modeling_auto,
    modeling_outputs,
)
from .module import (
    BaseHeadConfig,
    ContactPredictionHead,
    Criterion,
    HeadConfig,
    HeadRegistry,
    HeadTransformRegistry,
    IdentityTransform,
    LinearTransform,
    MaskedLMHead,
    MaskedLMHeadConfig,
    NonLinearTransform,
    PositionEmbeddingRegistry,
    PredictionHead,
    RotaryEmbedding,
    SequencePredictionHead,
    SinusoidalEmbedding,
    TokenKMerHead,
    TokenPredictionHead,
)
from .pipelines import RnaSecondaryStructurePipeline
from .tasks import Task, TaskLevel, TaskType
from .tokenisers import Alphabet, DnaTokenizer, DotBracketTokenizer, ProteinTokenizer, RnaTokenizer, Tokenizer
from .utils import count_parameters

__all__ = [
    "train",
    "evaluate",
    "infer",
    "modeling_auto",
    "modeling_outputs",
    "Dataset",
    "MultiMoleculeConfig",
    "MultiMoleculeRunner",
    "PreTrainedConfig",
    "HeadConfig",
    "BaseHeadConfig",
    "MaskedLMHeadConfig",
    "DnaTokenizer",
    "RnaTokenizer",
    "ProteinTokenizer",
    "DotBracketTokenizer",
    "Alphabet",
    "Tokenizer",
    "models",
    "AutoModelForContactPrediction",
    "AutoModelForNucleotidePrediction",
    "AutoModelForSequencePrediction",
    "AutoModelForTokenPrediction",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForContactPrediction",
    "CaLmForNucleotidePrediction",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "ErnieRnaConfig",
    "ErnieRnaModel",
    "ErnieRnaForContactPrediction",
    "ErnieRnaForNucleotidePrediction",
    "ErnieRnaForSequencePrediction",
    "ErnieRnaForTokenPrediction",
    "ErnieRnaForMaskedLM",
    "ErnieRnaForPreTraining",
    "ErnieRnaForSecondaryStructurePrediction",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoForContactPrediction",
    "RiNALMoForNucleotidePrediction",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForContactPrediction",
    "RnaBertForNucleotidePrediction",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaErnieConfig",
    "RnaErnieModel",
    "RnaErnieForContactPrediction",
    "RnaErnieForNucleotidePrediction",
    "RnaErnieForSequencePrediction",
    "RnaErnieForTokenPrediction",
    "RnaErnieForMaskedLM",
    "RnaErnieForPreTraining",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForContactPrediction",
    "RnaFmForNucleotidePrediction",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForContactPrediction",
    "RnaMsmForNucleotidePrediction",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForContactPrediction",
    "SpliceBertForNucleotidePrediction",
    "SpliceBertForSequencePrediction",
    "SpliceBertForTokenPrediction",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForContactPrediction",
    "UtrBertForNucleotidePrediction",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForContactPrediction",
    "UtrLmForNucleotidePrediction",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "HeadRegistry",
    "PredictionHead",
    "SequencePredictionHead",
    "TokenPredictionHead",
    "TokenKMerHead",
    "ContactPredictionHead",
    "MaskedLMHead",
    "HeadTransformRegistry",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
    "PositionEmbeddingRegistry",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "Criterion",
    "RnaSecondaryStructurePipeline",
    "count_parameters",
    "Task",
    "TaskLevel",
    "TaskType",
]
