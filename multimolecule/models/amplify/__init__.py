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
from .configuration_amplify import AMPLIFYConfig
from .modeling_amplify import (
    AMPLIFYForContactPrediction,
    AMPLIFYForMaskedLM,
    AMPLIFYForPreTraining,
    AMPLIFYForSequencePrediction,
    AMPLIFYForTokenPrediction,
    AMPLIFYModel,
    AMPLIFYPreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "AMPLIFYConfig",
    "AMPLIFYModel",
    "AMPLIFYPreTrainedModel",
    "AMPLIFYForContactPrediction",
    "AMPLIFYForSequencePrediction",
    "AMPLIFYForTokenPrediction",
    "AMPLIFYForMaskedLM",
    "AMPLIFYForPreTraining",
]

AutoConfig.register("amplify", AMPLIFYConfig)
AutoBackbone.register(AMPLIFYConfig, AMPLIFYModel)
AutoModel.register(AMPLIFYConfig, AMPLIFYModel)
AutoModelForContactPrediction.register(AMPLIFYConfig, AMPLIFYForContactPrediction)
AutoModelForSequencePrediction.register(AMPLIFYConfig, AMPLIFYForSequencePrediction)
AutoModelForSequenceClassification.register(AMPLIFYConfig, AMPLIFYForSequencePrediction)
AutoModelForTokenPrediction.register(AMPLIFYConfig, AMPLIFYForTokenPrediction)
AutoModelForTokenClassification.register(AMPLIFYConfig, AMPLIFYForTokenPrediction)
AutoModelForMaskedLM.register(AMPLIFYConfig, AMPLIFYForMaskedLM)
AutoModelForPreTraining.register(AMPLIFYConfig, AMPLIFYForPreTraining)
AutoTokenizer.register(AMPLIFYConfig, ProteinTokenizer)
