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
from .configuration_amplify import AmplifyConfig
from .modeling_amplify import (
    AmplifyForContactPrediction,
    AmplifyForMaskedLM,
    AmplifyForPreTraining,
    AmplifyForSequencePrediction,
    AmplifyForTokenPrediction,
    AmplifyModel,
    AmplifyPreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "AmplifyConfig",
    "AmplifyModel",
    "AmplifyPreTrainedModel",
    "AmplifyForContactPrediction",
    "AmplifyForSequencePrediction",
    "AmplifyForTokenPrediction",
    "AmplifyForMaskedLM",
    "AmplifyForPreTraining",
]

AutoConfig.register("amplify", AmplifyConfig)
AutoBackbone.register(AmplifyConfig, AmplifyModel)
AutoModel.register(AmplifyConfig, AmplifyModel)
AutoModelForContactPrediction.register(AmplifyConfig, AmplifyForContactPrediction)
AutoModelForSequencePrediction.register(AmplifyConfig, AmplifyForSequencePrediction)
AutoModelForSequenceClassification.register(AmplifyConfig, AmplifyForSequencePrediction)
AutoModelForTokenPrediction.register(AmplifyConfig, AmplifyForTokenPrediction)
AutoModelForTokenClassification.register(AmplifyConfig, AmplifyForTokenPrediction)
AutoModelForMaskedLM.register(AmplifyConfig, AmplifyForMaskedLM)
AutoModelForPreTraining.register(AmplifyConfig, AmplifyForPreTraining)
AutoTokenizer.register(AmplifyConfig, ProteinTokenizer)
