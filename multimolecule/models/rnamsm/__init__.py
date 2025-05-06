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
from .configuration_rnamsm import RnaMsmConfig
from .modeling_rnamsm import (
    RnaMsmForContactPrediction,
    RnaMsmForMaskedLM,
    RnaMsmForPreTraining,
    RnaMsmForSecondaryStructurePrediction,
    RnaMsmForSequencePrediction,
    RnaMsmForTokenPrediction,
    RnaMsmModel,
    RnaMsmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmPreTrainedModel",
    "RnaMsmForContactPrediction",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "RnaMsmForSecondaryStructurePrediction",
]

AutoConfig.register("rnamsm", RnaMsmConfig)
AutoBackbone.register(RnaMsmConfig, RnaMsmModel)
AutoModel.register(RnaMsmConfig, RnaMsmModel)
AutoModelForContactPrediction.register(RnaMsmConfig, RnaMsmForContactPrediction)
AutoModelForSequencePrediction.register(RnaMsmConfig, RnaMsmForSequencePrediction)
AutoModelForSequenceClassification.register(RnaMsmConfig, RnaMsmForSequencePrediction)
AutoModelForTokenPrediction.register(RnaMsmConfig, RnaMsmForTokenPrediction)
AutoModelForTokenClassification.register(RnaMsmConfig, RnaMsmForTokenPrediction)
AutoModelForMaskedLM.register(RnaMsmConfig, RnaMsmForMaskedLM)
AutoModelForPreTraining.register(RnaMsmConfig, RnaMsmForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(RnaMsmConfig, RnaMsmForSecondaryStructurePrediction)
AutoTokenizer.register(RnaMsmConfig, RnaTokenizer)
