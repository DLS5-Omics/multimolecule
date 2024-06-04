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
from .configuration_ernierna import ErnieRnaConfig
from .modeling_ernierna import (
    ErnieRnaForContactClassification,
    ErnieRnaForContactPrediction,
    ErnieRnaForMaskedLM,
    ErnieRnaForNucleotidePrediction,
    ErnieRnaForPreTraining,
    ErnieRnaForSequencePrediction,
    ErnieRnaForTokenPrediction,
    ErnieRnaModel,
    ErnieRnaPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "ErnieRnaConfig",
    "ErnieRnaModel",
    "ErnieRnaPreTrainedModel",
    "ErnieRnaForContactPrediction",
    "ErnieRnaForNucleotidePrediction",
    "ErnieRnaForSequencePrediction",
    "ErnieRnaForTokenPrediction",
    "ErnieRnaForMaskedLM",
    "ErnieRnaForPreTraining",
    "ErnieRnaForContactClassification",
]

AutoConfig.register("ernierna", ErnieRnaConfig)
AutoBackbone.register(ErnieRnaConfig, ErnieRnaModel)
AutoModel.register(ErnieRnaConfig, ErnieRnaModel)
AutoModelForContactPrediction.register(ErnieRnaConfig, ErnieRnaForContactPrediction)
AutoModelForNucleotidePrediction.register(ErnieRnaConfig, ErnieRnaForNucleotidePrediction)
AutoModelForSequencePrediction.register(ErnieRnaConfig, ErnieRnaForSequencePrediction)
AutoModelForSequenceClassification.register(ErnieRnaConfig, ErnieRnaForSequencePrediction)
AutoModelForTokenPrediction.register(ErnieRnaConfig, ErnieRnaForTokenPrediction)
AutoModelForTokenClassification.register(ErnieRnaConfig, ErnieRnaForTokenPrediction)
AutoModelForMaskedLM.register(ErnieRnaConfig, ErnieRnaForMaskedLM)
AutoModelForPreTraining.register(ErnieRnaConfig, ErnieRnaForPreTraining)
AutoTokenizer.register(ErnieRnaConfig, RnaTokenizer)
