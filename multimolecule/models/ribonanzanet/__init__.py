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
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers.rna import RnaTokenizer

from ..modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForRnaSecondaryStructurePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .configuration_ribonanzanet import RibonanzaNetConfig
from .modeling_ribonanzanet import (
    RibonanzaNetForContactPrediction,
    RibonanzaNetForDegradationPrediction,
    RibonanzaNetForPreTraining,
    RibonanzaNetForSecondaryStructurePrediction,
    RibonanzaNetForSequenceDropoutPrediction,
    RibonanzaNetForSequencePrediction,
    RibonanzaNetForTokenPrediction,
    RibonanzaNetModel,
    RibonanzaNetPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RibonanzaNetConfig",
    "RibonanzaNetModel",
    "RibonanzaNetPreTrainedModel",
    "RibonanzaNetForContactPrediction",
    "RibonanzaNetForSequencePrediction",
    "RibonanzaNetForTokenPrediction",
    "RibonanzaNetForPreTraining",
    "RibonanzaNetForSecondaryStructurePrediction",
    "RibonanzaNetForDegradationPrediction",
    "RibonanzaNetForSequenceDropoutPrediction",
]

AutoConfig.register("ribonanzanet", RibonanzaNetConfig)
AutoBackbone.register(RibonanzaNetConfig, RibonanzaNetModel)
AutoModel.register(RibonanzaNetConfig, RibonanzaNetModel)
AutoModelForContactPrediction.register(RibonanzaNetConfig, RibonanzaNetForContactPrediction)
AutoModelForSequencePrediction.register(RibonanzaNetConfig, RibonanzaNetForSequencePrediction)
AutoModelForSequenceClassification.register(RibonanzaNetConfig, RibonanzaNetForSequencePrediction)
AutoModelForTokenPrediction.register(RibonanzaNetConfig, RibonanzaNetForTokenPrediction)
AutoModelForTokenClassification.register(RibonanzaNetConfig, RibonanzaNetForTokenPrediction)
AutoModelForPreTraining.register(RibonanzaNetConfig, RibonanzaNetForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(RibonanzaNetConfig, RibonanzaNetForSecondaryStructurePrediction)
AutoTokenizer.register(RibonanzaNetConfig, RnaTokenizer)
