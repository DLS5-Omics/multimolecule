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

from ..modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .configuration_ablang2 import AbLang2Config
from .modeling_ablang2 import (
    AbLang2ForContactPrediction,
    AbLang2ForMaskedLM,
    AbLang2ForPreTraining,
    AbLang2ForSequencePrediction,
    AbLang2ForTokenPrediction,
    AbLang2Model,
    AbLang2PreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "AbLang2Config",
    "AbLang2Model",
    "AbLang2PreTrainedModel",
    "AbLang2ForContactPrediction",
    "AbLang2ForSequencePrediction",
    "AbLang2ForTokenPrediction",
    "AbLang2ForMaskedLM",
    "AbLang2ForPreTraining",
]

AutoConfig.register("ablang2", AbLang2Config)
AutoBackbone.register(AbLang2Config, AbLang2Model)
AutoModel.register(AbLang2Config, AbLang2Model)
AutoModelForContactPrediction.register(AbLang2Config, AbLang2ForContactPrediction)
AutoModelForSequencePrediction.register(AbLang2Config, AbLang2ForSequencePrediction)
AutoModelForSequenceClassification.register(AbLang2Config, AbLang2ForSequencePrediction)
AutoModelForTokenPrediction.register(AbLang2Config, AbLang2ForTokenPrediction)
AutoModelForTokenClassification.register(AbLang2Config, AbLang2ForTokenPrediction)
AutoModelForMaskedLM.register(AbLang2Config, AbLang2ForMaskedLM)
AutoModelForPreTraining.register(AbLang2Config, AbLang2ForPreTraining)
AutoTokenizer.register(AbLang2Config, ProteinTokenizer)
