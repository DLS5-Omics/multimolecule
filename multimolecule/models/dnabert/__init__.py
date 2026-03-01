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

from multimolecule.tokenisers import DnaTokenizer

from ..modeling_auto import AutoModelForContactPrediction, AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_dnabert import DnaBertConfig
from .modeling_dnabert import (
    DnaBertForContactPrediction,
    DnaBertForMaskedLM,
    DnaBertForPreTraining,
    DnaBertForSequencePrediction,
    DnaBertForTokenPrediction,
    DnaBertModel,
    DnaBertPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "DnaBertConfig",
    "DnaBertModel",
    "DnaBertPreTrainedModel",
    "DnaBertForContactPrediction",
    "DnaBertForSequencePrediction",
    "DnaBertForTokenPrediction",
    "DnaBertForMaskedLM",
    "DnaBertForPreTraining",
]

AutoConfig.register("dnabert", DnaBertConfig)
AutoBackbone.register(DnaBertConfig, DnaBertModel)
AutoModel.register(DnaBertConfig, DnaBertModel)
AutoModelForContactPrediction.register(DnaBertConfig, DnaBertForContactPrediction)
AutoModelForSequencePrediction.register(DnaBertConfig, DnaBertForSequencePrediction)
AutoModelForSequenceClassification.register(DnaBertConfig, DnaBertForSequencePrediction)
AutoModelForTokenPrediction.register(DnaBertConfig, DnaBertForTokenPrediction)
AutoModelForTokenClassification.register(DnaBertConfig, DnaBertForTokenPrediction)
AutoModelForMaskedLM.register(DnaBertConfig, DnaBertForMaskedLM)
AutoModelForPreTraining.register(DnaBertConfig, DnaBertForPreTraining)
AutoTokenizer.register(DnaBertConfig, DnaTokenizer)
