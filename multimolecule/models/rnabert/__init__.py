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
from .configuration_rnabert import RnaBertConfig
from .modeling_rnabert import (
    RnaBertForContactPrediction,
    RnaBertForMaskedLM,
    RnaBertForNucleotidePrediction,
    RnaBertForPreTraining,
    RnaBertForSequencePrediction,
    RnaBertForTokenPrediction,
    RnaBertModel,
    RnaBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertPreTrainedModel",
    "RnaBertForContactPrediction",
    "RnaBertForNucleotidePrediction",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
]

AutoConfig.register("rnabert", RnaBertConfig)
AutoBackbone.register(RnaBertConfig, RnaBertModel)
AutoModel.register(RnaBertConfig, RnaBertModel)
AutoModelForContactPrediction.register(RnaBertConfig, RnaBertForContactPrediction)
AutoModelForNucleotidePrediction.register(RnaBertConfig, RnaBertForNucleotidePrediction)
AutoModelForSequencePrediction.register(RnaBertConfig, RnaBertForSequencePrediction)
AutoModelForSequenceClassification.register(RnaBertConfig, RnaBertForSequencePrediction)
AutoModelForTokenPrediction.register(RnaBertConfig, RnaBertForTokenPrediction)
AutoModelForTokenClassification.register(RnaBertConfig, RnaBertForTokenPrediction)
AutoModelForMaskedLM.register(RnaBertConfig, RnaBertForMaskedLM)
AutoModelForPreTraining.register(RnaBertConfig, RnaBertForPreTraining)
AutoTokenizer.register(RnaBertConfig, RnaTokenizer)
