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
from .configuration_esmc import EsmCConfig
from .modeling_esmc import (
    EsmCForContactPrediction,
    EsmCForMaskedLM,
    EsmCForPreTraining,
    EsmCForSequencePrediction,
    EsmCForTokenPrediction,
    EsmCModel,
    EsmCPreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "EsmCConfig",
    "EsmCModel",
    "EsmCPreTrainedModel",
    "EsmCForContactPrediction",
    "EsmCForSequencePrediction",
    "EsmCForTokenPrediction",
    "EsmCForMaskedLM",
    "EsmCForPreTraining",
]

AutoConfig.register("esmc", EsmCConfig)
AutoBackbone.register(EsmCConfig, EsmCModel)
AutoModel.register(EsmCConfig, EsmCModel)
AutoModelForContactPrediction.register(EsmCConfig, EsmCForContactPrediction)
AutoModelForSequencePrediction.register(EsmCConfig, EsmCForSequencePrediction)
AutoModelForSequenceClassification.register(EsmCConfig, EsmCForSequencePrediction)
AutoModelForTokenPrediction.register(EsmCConfig, EsmCForTokenPrediction)
AutoModelForTokenClassification.register(EsmCConfig, EsmCForTokenPrediction)
AutoModelForMaskedLM.register(EsmCConfig, EsmCForMaskedLM)
AutoModelForPreTraining.register(EsmCConfig, EsmCForPreTraining)
AutoTokenizer.register(EsmCConfig, ProteinTokenizer)
