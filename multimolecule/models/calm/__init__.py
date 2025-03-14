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
from .configuration_calm import CaLmConfig
from .modeling_calm import (
    CaLmForContactPrediction,
    CaLmForMaskedLM,
    CaLmForPreTraining,
    CaLmForSequencePrediction,
    CaLmForTokenPrediction,
    CaLmModel,
    CaLmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "CaLmConfig",
    "CaLmModel",
    "CaLmPreTrainedModel",
    "CaLmForContactPrediction",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
]

AutoConfig.register("calm", CaLmConfig)
AutoBackbone.register(CaLmConfig, CaLmModel)
AutoModel.register(CaLmConfig, CaLmModel)
AutoModelForContactPrediction.register(CaLmConfig, CaLmForContactPrediction)
AutoModelForSequencePrediction.register(CaLmConfig, CaLmForSequencePrediction)
AutoModelForSequenceClassification.register(CaLmConfig, CaLmForSequencePrediction)
AutoModelForTokenPrediction.register(CaLmConfig, CaLmForTokenPrediction)
AutoModelForTokenClassification.register(CaLmConfig, CaLmForTokenPrediction)
AutoModelForMaskedLM.register(CaLmConfig, CaLmForMaskedLM)
AutoModelForPreTraining.register(CaLmConfig, CaLmForPreTraining)
AutoTokenizer.register(CaLmConfig, DnaTokenizer)
