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


from .apis import evaluate, inference, train
from .data import Dataset, contact_map_to_dot_bracket, dot_bracket_to_contact_map
from .models import (
    AutoModelForContactPrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
    CaLmConfig,
    CaLmForContactPrediction,
    CaLmForMaskedLM,
    CaLmForPreTraining,
    CaLmForSequencePrediction,
    CaLmForTokenPrediction,
    CaLmModel,
    ErnieRnaConfig,
    ErnieRnaForContactPrediction,
    ErnieRnaForMaskedLM,
    ErnieRnaForPreTraining,
    ErnieRnaForSecondaryStructurePrediction,
    ErnieRnaForSequencePrediction,
    ErnieRnaForTokenPrediction,
    ErnieRnaModel,
    PreTrainedConfig,
    RibonanzaNetConfig,
    RibonanzaNetForContactPrediction,
    RibonanzaNetForDegradationPrediction,
    RibonanzaNetForPreTraining,
    RibonanzaNetForSecondaryStructurePrediction,
    RibonanzaNetForSequenceDropoutPrediction,
    RibonanzaNetForSequencePrediction,
    RibonanzaNetForTokenPrediction,
    RibonanzaNetModel,
    RiNALMoConfig,
    RiNALMoForContactPrediction,
    RiNALMoForMaskedLM,
    RiNALMoForPreTraining,
    RiNALMoForSequencePrediction,
    RiNALMoForTokenPrediction,
    RiNALMoModel,
    RnaBertConfig,
    RnaBertForContactPrediction,
    RnaBertForMaskedLM,
    RnaBertForPreTraining,
    RnaBertForSequencePrediction,
    RnaBertForTokenPrediction,
    RnaBertModel,
    RnaErnieConfig,
    RnaErnieForContactPrediction,
    RnaErnieForMaskedLM,
    RnaErnieForPreTraining,
    RnaErnieForSequencePrediction,
    RnaErnieForTokenPrediction,
    RnaErnieModel,
    RnaFmConfig,
    RnaFmForContactPrediction,
    RnaFmForMaskedLM,
    RnaFmForPreTraining,
    RnaFmForSecondaryStructurePrediction,
    RnaFmForSequencePrediction,
    RnaFmForTokenPrediction,
    RnaFmModel,
    RnaMsmConfig,
    RnaMsmForContactPrediction,
    RnaMsmForMaskedLM,
    RnaMsmForPreTraining,
    RnaMsmForSecondaryStructurePrediction,
    RnaMsmForSequencePrediction,
    RnaMsmForTokenPrediction,
    RnaMsmModel,
    SpliceBertConfig,
    SpliceBertForContactPrediction,
    SpliceBertForMaskedLM,
    SpliceBertForPreTraining,
    SpliceBertForSequencePrediction,
    SpliceBertForTokenPrediction,
    SpliceBertModel,
    UtrBertConfig,
    UtrBertForContactPrediction,
    UtrBertForMaskedLM,
    UtrBertForPreTraining,
    UtrBertForSequencePrediction,
    UtrBertForTokenPrediction,
    UtrBertModel,
    UtrLmConfig,
    UtrLmForContactPrediction,
    UtrLmForMaskedLM,
    UtrLmForPreTraining,
    UtrLmForSecondaryStructurePrediction,
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
    HeadTransformRegistryHF,
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
from .runner import Config, Runner
from .tasks import Task, TaskLevel, TaskType
from .tokenisers import Alphabet, DnaTokenizer, DotBracketTokenizer, ProteinTokenizer, RnaTokenizer, Tokenizer
from .utils import count_parameters

__all__ = [
    "train",
    "evaluate",
    "inference",
    "modeling_auto",
    "modeling_outputs",
    "Dataset",
    "dot_bracket_to_contact_map",
    "contact_map_to_dot_bracket",
    "Config",
    "Runner",
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
    "AutoModelForSequencePrediction",
    "AutoModelForTokenPrediction",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForContactPrediction",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "ErnieRnaConfig",
    "ErnieRnaModel",
    "ErnieRnaForContactPrediction",
    "ErnieRnaForSequencePrediction",
    "ErnieRnaForTokenPrediction",
    "ErnieRnaForMaskedLM",
    "ErnieRnaForPreTraining",
    "ErnieRnaForSecondaryStructurePrediction",
    "RibonanzaNetConfig",
    "RibonanzaNetModel",
    "RibonanzaNetForContactPrediction",
    "RibonanzaNetForSequencePrediction",
    "RibonanzaNetForTokenPrediction",
    "RibonanzaNetForPreTraining",
    "RibonanzaNetForSecondaryStructurePrediction",
    "RibonanzaNetForDegradationPrediction",
    "RibonanzaNetForSequenceDropoutPrediction",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoForContactPrediction",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForContactPrediction",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaErnieConfig",
    "RnaErnieModel",
    "RnaErnieForContactPrediction",
    "RnaErnieForSequencePrediction",
    "RnaErnieForTokenPrediction",
    "RnaErnieForMaskedLM",
    "RnaErnieForPreTraining",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForContactPrediction",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaFmForSecondaryStructurePrediction",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForContactPrediction",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "RnaMsmForSecondaryStructurePrediction",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForContactPrediction",
    "SpliceBertForSequencePrediction",
    "SpliceBertForTokenPrediction",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForContactPrediction",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForContactPrediction",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "UtrLmForSecondaryStructurePrediction",
    "HeadRegistry",
    "PredictionHead",
    "SequencePredictionHead",
    "TokenPredictionHead",
    "TokenKMerHead",
    "ContactPredictionHead",
    "MaskedLMHead",
    "HeadTransformRegistry",
    "HeadTransformRegistryHF",
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
