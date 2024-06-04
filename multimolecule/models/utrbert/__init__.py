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

from multimolecule.tokenisers.rna import RnaTokenizer

from ..modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForNucleotidePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .configuration_utrbert import UtrBertConfig
from .modeling_utrbert import (
    UtrBertForContactPrediction,
    UtrBertForMaskedLM,
    UtrBertForNucleotidePrediction,
    UtrBertForPreTraining,
    UtrBertForSequencePrediction,
    UtrBertForTokenPrediction,
    UtrBertModel,
    UtrBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertPreTrainedModel",
    "UtrBertForContactPrediction",
    "UtrBertForNucleotidePrediction",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
]

AutoConfig.register("utrbert", UtrBertConfig)
AutoBackbone.register(UtrBertConfig, UtrBertModel)
AutoModel.register(UtrBertConfig, UtrBertModel)
AutoModelForContactPrediction.register(UtrBertConfig, UtrBertForContactPrediction)
AutoModelForNucleotidePrediction.register(UtrBertConfig, UtrBertForNucleotidePrediction)
AutoModelForSequencePrediction.register(UtrBertConfig, UtrBertForSequencePrediction)
AutoModelForSequenceClassification.register(UtrBertConfig, UtrBertForSequencePrediction)
AutoModelForTokenPrediction.register(UtrBertConfig, UtrBertForTokenPrediction)
AutoModelForTokenClassification.register(UtrBertConfig, UtrBertForTokenPrediction)
AutoModelForMaskedLM.register(UtrBertConfig, UtrBertForMaskedLM)
AutoModelForPreTraining.register(UtrBertConfig, UtrBertForPreTraining)
AutoTokenizer.register(UtrBertConfig, RnaTokenizer)
