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

from ..modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForRnaSecondaryStructurePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .configuration_utrlm import UtrLmConfig
from .modeling_utrlm import (
    UtrLmForContactPrediction,
    UtrLmForMaskedLM,
    UtrLmForPreTraining,
    UtrLmForSecondaryStructurePrediction,
    UtrLmForSequencePrediction,
    UtrLmForTokenPrediction,
    UtrLmModel,
    UtrLmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmPreTrainedModel",
    "UtrLmForContactPrediction",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "UtrLmForSecondaryStructurePrediction",
]

AutoConfig.register("utrlm", UtrLmConfig)
AutoBackbone.register(UtrLmConfig, UtrLmModel)
AutoModel.register(UtrLmConfig, UtrLmModel)
AutoModelForContactPrediction.register(UtrLmConfig, UtrLmForContactPrediction)
AutoModelForSequencePrediction.register(UtrLmConfig, UtrLmForSequencePrediction)
AutoModelForSequenceClassification.register(UtrLmConfig, UtrLmForSequencePrediction)
AutoModelForTokenPrediction.register(UtrLmConfig, UtrLmForTokenPrediction)
AutoModelForTokenClassification.register(UtrLmConfig, UtrLmForTokenPrediction)
AutoModelForMaskedLM.register(UtrLmConfig, UtrLmForMaskedLM)
AutoModelForPreTraining.register(UtrLmConfig, UtrLmForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(UtrLmConfig, UtrLmForSecondaryStructurePrediction)
AutoTokenizer.register(UtrLmConfig, RnaTokenizer)
