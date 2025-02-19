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
from .configuration_rnafm import RnaFmConfig
from .modeling_rnafm import (
    RnaFmForContactPrediction,
    RnaFmForMaskedLM,
    RnaFmForPreTraining,
    RnaFmForSecondaryStructurePrediction,
    RnaFmForSequencePrediction,
    RnaFmForTokenPrediction,
    RnaFmModel,
    RnaFmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmPreTrainedModel",
    "RnaFmForContactPrediction",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaFmForSecondaryStructurePrediction",
]

AutoConfig.register("rnafm", RnaFmConfig)
AutoBackbone.register(RnaFmConfig, RnaFmModel)
AutoModel.register(RnaFmConfig, RnaFmModel)
AutoModelForContactPrediction.register(RnaFmConfig, RnaFmForContactPrediction)
AutoModelForSequencePrediction.register(RnaFmConfig, RnaFmForSequencePrediction)
AutoModelForSequenceClassification.register(RnaFmConfig, RnaFmForSequencePrediction)
AutoModelForTokenPrediction.register(RnaFmConfig, RnaFmForTokenPrediction)
AutoModelForTokenClassification.register(RnaFmConfig, RnaFmForTokenPrediction)
AutoModelForMaskedLM.register(RnaFmConfig, RnaFmForMaskedLM)
AutoModelForPreTraining.register(RnaFmConfig, RnaFmForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(RnaFmConfig, RnaFmForSecondaryStructurePrediction)
AutoTokenizer.register(RnaFmConfig, RnaTokenizer)
