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
from .configuration_generanno import GenerannoConfig
from .modeling_generanno import (
    GenerannoForContactPrediction,
    GenerannoForMaskedLM,
    GenerannoForPreTraining,
    GenerannoForSequencePrediction,
    GenerannoForTokenPrediction,
    GenerannoModel,
    GenerannoPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "GenerannoConfig",
    "GenerannoModel",
    "GenerannoPreTrainedModel",
    "GenerannoForContactPrediction",
    "GenerannoForSequencePrediction",
    "GenerannoForTokenPrediction",
    "GenerannoForMaskedLM",
    "GenerannoForPreTraining",
]

AutoConfig.register("generanno", GenerannoConfig)
AutoBackbone.register(GenerannoConfig, GenerannoModel)
AutoModel.register(GenerannoConfig, GenerannoModel)
AutoModelForContactPrediction.register(GenerannoConfig, GenerannoForContactPrediction)
AutoModelForSequencePrediction.register(GenerannoConfig, GenerannoForSequencePrediction)
AutoModelForSequenceClassification.register(GenerannoConfig, GenerannoForSequencePrediction)
AutoModelForTokenPrediction.register(GenerannoConfig, GenerannoForTokenPrediction)
AutoModelForTokenClassification.register(GenerannoConfig, GenerannoForTokenPrediction)
AutoModelForMaskedLM.register(GenerannoConfig, GenerannoForMaskedLM)
AutoModelForPreTraining.register(GenerannoConfig, GenerannoForPreTraining)
AutoTokenizer.register(GenerannoConfig, DnaTokenizer)
