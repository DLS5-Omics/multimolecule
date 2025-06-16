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

from multimolecule.tokenisers import RnaTokenizer

from ..modeling_auto import AutoModelForContactPrediction, AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_ncrnabert import NcRnaBertConfig
from .modeling_ncrnabert import (
    NcRnaBertForContactPrediction,
    NcRnaBertForMaskedLM,
    NcRnaBertForPreTraining,
    NcRnaBertForSequencePrediction,
    NcRnaBertForTokenPrediction,
    NcRnaBertModel,
    NcRnaBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "NcRnaBertConfig",
    "NcRnaBertModel",
    "NcRnaBertPreTrainedModel",
    "NcRnaBertForContactPrediction",
    "NcRnaBertForSequencePrediction",
    "NcRnaBertForTokenPrediction",
    "NcRnaBertForMaskedLM",
    "NcRnaBertForPreTraining",
]

AutoConfig.register("ncrnabert", NcRnaBertConfig)
AutoBackbone.register(NcRnaBertConfig, NcRnaBertModel)
AutoModel.register(NcRnaBertConfig, NcRnaBertModel)
AutoModelForContactPrediction.register(NcRnaBertConfig, NcRnaBertForContactPrediction)
AutoModelForSequencePrediction.register(NcRnaBertConfig, NcRnaBertForSequencePrediction)
AutoModelForSequenceClassification.register(NcRnaBertConfig, NcRnaBertForSequencePrediction)
AutoModelForTokenPrediction.register(NcRnaBertConfig, NcRnaBertForTokenPrediction)
AutoModelForTokenClassification.register(NcRnaBertConfig, NcRnaBertForTokenPrediction)
AutoModelForMaskedLM.register(NcRnaBertConfig, NcRnaBertForMaskedLM)
AutoModelForPreTraining.register(NcRnaBertConfig, NcRnaBertForPreTraining)
AutoTokenizer.register(NcRnaBertConfig, RnaTokenizer)
