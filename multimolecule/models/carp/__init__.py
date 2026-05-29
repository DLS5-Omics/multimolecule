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

from ..modeling_auto import AutoModelForContactPrediction, AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_carp import CarpConfig
from .modeling_carp import (
    CarpEncoderOutput,
    CarpForContactPrediction,
    CarpForMaskedLM,
    CarpForPreTraining,
    CarpForSequencePrediction,
    CarpForTokenPrediction,
    CarpModel,
    CarpModelOutput,
    CarpPreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "CarpConfig",
    "CarpModel",
    "CarpPreTrainedModel",
    "CarpModelOutput",
    "CarpEncoderOutput",
    "CarpForSequencePrediction",
    "CarpForTokenPrediction",
    "CarpForContactPrediction",
    "CarpForMaskedLM",
    "CarpForPreTraining",
]

AutoConfig.register("carp", CarpConfig)
AutoBackbone.register(CarpConfig, CarpModel)
AutoModel.register(CarpConfig, CarpModel)
AutoModelForSequencePrediction.register(CarpConfig, CarpForSequencePrediction)
AutoModelForSequenceClassification.register(CarpConfig, CarpForSequencePrediction)
AutoModelForTokenPrediction.register(CarpConfig, CarpForTokenPrediction)
AutoModelForTokenClassification.register(CarpConfig, CarpForTokenPrediction)
AutoModelForContactPrediction.register(CarpConfig, CarpForContactPrediction)
AutoModelForMaskedLM.register(CarpConfig, CarpForMaskedLM)
AutoModelForPreTraining.register(CarpConfig, CarpForPreTraining)
AutoTokenizer.register(CarpConfig, ProteinTokenizer)
