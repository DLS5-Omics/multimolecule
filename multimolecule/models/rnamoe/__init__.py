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
from .configuration_rnamoe import RnaMoeConfig
from .modeling_rnamoe import (
    RnaMoeForContactPrediction,
    RnaMoeForDrugImprovement,
    RnaMoeForMaskedLM,
    RnaMoeForPreTraining,
    RnaMoeForSecondaryStructurePrediction,
    RnaMoeForSequencePrediction,
    RnaMoeForTokenPrediction,
    RnaMoeModel,
    RnaMoePreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaMoeConfig",
    "RnaMoeModel",
    "RnaMoePreTrainedModel",
    "RnaMoeForContactPrediction",
    "RnaMoeForSequencePrediction",
    "RnaMoeForTokenPrediction",
    "RnaMoeForMaskedLM",
    "RnaMoeForPreTraining",
    "RnaMoeForSecondaryStructurePrediction",
    "RnaMoeForDrugImprovement",
]

AutoConfig.register("rnamoe", RnaMoeConfig)
AutoBackbone.register(RnaMoeConfig, RnaMoeModel)
AutoModel.register(RnaMoeConfig, RnaMoeModel)
AutoModelForContactPrediction.register(RnaMoeConfig, RnaMoeForContactPrediction)
AutoModelForSequencePrediction.register(RnaMoeConfig, RnaMoeForSequencePrediction)
AutoModelForSequenceClassification.register(RnaMoeConfig, RnaMoeForSequencePrediction)
AutoModelForTokenPrediction.register(RnaMoeConfig, RnaMoeForTokenPrediction)
AutoModelForTokenClassification.register(RnaMoeConfig, RnaMoeForTokenPrediction)
AutoModelForMaskedLM.register(RnaMoeConfig, RnaMoeForMaskedLM)
AutoModelForPreTraining.register(RnaMoeConfig, RnaMoeForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(RnaMoeConfig, RnaMoeForSecondaryStructurePrediction)
AutoTokenizer.register(RnaMoeConfig, RnaTokenizer)
