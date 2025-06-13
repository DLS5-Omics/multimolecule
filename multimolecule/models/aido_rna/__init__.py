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
from .configuration_aido_rna import AidoRnaConfig
from .modeling_aido_rna import (
    AidoRnaForContactPrediction,
    AidoRnaForMaskedLM,
    AidoRnaForPreTraining,
    AidoRnaForSecondaryStructurePrediction,
    AidoRnaForSequencePrediction,
    AidoRnaForTokenPrediction,
    AidoRnaModel,
    AidoRnaPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "AidoRnaConfig",
    "AidoRnaModel",
    "AidoRnaPreTrainedModel",
    "AidoRnaForContactPrediction",
    "AidoRnaForSequencePrediction",
    "AidoRnaForTokenPrediction",
    "AidoRnaForMaskedLM",
    "AidoRnaForPreTraining",
    "AidoRnaForSecondaryStructurePrediction",
]

AutoConfig.register("aido.rna", AidoRnaConfig)
AutoBackbone.register(AidoRnaConfig, AidoRnaModel)
AutoModel.register(AidoRnaConfig, AidoRnaModel)
AutoModelForContactPrediction.register(AidoRnaConfig, AidoRnaForContactPrediction)
AutoModelForSequencePrediction.register(AidoRnaConfig, AidoRnaForSequencePrediction)
AutoModelForSequenceClassification.register(AidoRnaConfig, AidoRnaForSequencePrediction)
AutoModelForTokenPrediction.register(AidoRnaConfig, AidoRnaForTokenPrediction)
AutoModelForTokenClassification.register(AidoRnaConfig, AidoRnaForTokenPrediction)
AutoModelForMaskedLM.register(AidoRnaConfig, AidoRnaForMaskedLM)
AutoModelForPreTraining.register(AidoRnaConfig, AidoRnaForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(AidoRnaConfig, AidoRnaForSecondaryStructurePrediction)
AutoTokenizer.register(AidoRnaConfig, RnaTokenizer)
