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


from transformers import (
    AutoBackbone,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers import ProteinTokenizer

from ..modeling_auto import AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_proteinbert import ProteinBertConfig
from .modeling_proteinbert import (
    ProteinBertForMaskedLM,
    ProteinBertForPreTraining,
    ProteinBertForPreTrainingOutput,
    ProteinBertForSequencePrediction,
    ProteinBertForTokenPrediction,
    ProteinBertModel,
    ProteinBertModelOutput,
    ProteinBertPreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "ProteinBertConfig",
    "ProteinBertModel",
    "ProteinBertPreTrainedModel",
    "ProteinBertForSequencePrediction",
    "ProteinBertForTokenPrediction",
    "ProteinBertForMaskedLM",
    "ProteinBertForPreTraining",
    "ProteinBertModelOutput",
    "ProteinBertForPreTrainingOutput",
]

AutoConfig.register("proteinbert", ProteinBertConfig)
AutoBackbone.register(ProteinBertConfig, ProteinBertModel)
AutoModel.register(ProteinBertConfig, ProteinBertModel)
AutoModelForSequencePrediction.register(ProteinBertConfig, ProteinBertForSequencePrediction)
AutoModelForSequenceClassification.register(ProteinBertConfig, ProteinBertForSequencePrediction)
AutoModelForTokenPrediction.register(ProteinBertConfig, ProteinBertForTokenPrediction)
AutoModelForTokenClassification.register(ProteinBertConfig, ProteinBertForTokenPrediction)
AutoModelForMaskedLM.register(ProteinBertConfig, ProteinBertForMaskedLM)
AutoModelForPreTraining.register(ProteinBertConfig, ProteinBertForPreTraining)
AutoTokenizer.register(ProteinBertConfig, ProteinTokenizer)
