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
from .configuration_ablang import AbLangConfig
from .modeling_ablang import (
    AbLangForMaskedLM,
    AbLangForPreTraining,
    AbLangForSequencePrediction,
    AbLangForTokenPrediction,
    AbLangModel,
    AbLangPreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "AbLangConfig",
    "AbLangModel",
    "AbLangPreTrainedModel",
    "AbLangForSequencePrediction",
    "AbLangForTokenPrediction",
    "AbLangForMaskedLM",
    "AbLangForPreTraining",
]

AutoConfig.register("ablang", AbLangConfig)
AutoBackbone.register(AbLangConfig, AbLangModel)
AutoModel.register(AbLangConfig, AbLangModel)
AutoModelForSequencePrediction.register(AbLangConfig, AbLangForSequencePrediction)
AutoModelForSequenceClassification.register(AbLangConfig, AbLangForSequencePrediction)
AutoModelForTokenPrediction.register(AbLangConfig, AbLangForTokenPrediction)
AutoModelForTokenClassification.register(AbLangConfig, AbLangForTokenPrediction)
AutoModelForMaskedLM.register(AbLangConfig, AbLangForMaskedLM)
AutoModelForPreTraining.register(AbLangConfig, AbLangForPreTraining)
AutoTokenizer.register(AbLangConfig, ProteinTokenizer)
