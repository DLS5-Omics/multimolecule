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
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers import DnaTokenizer

from ..modeling_auto import AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_hyenadna import HyenaDnaConfig
from .modeling_hyenadna import (
    HyenaDnaForCausalLM,
    HyenaDnaForPreTraining,
    HyenaDnaForSequencePrediction,
    HyenaDnaForTokenPrediction,
    HyenaDnaModel,
    HyenaDnaPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "HyenaDnaConfig",
    "HyenaDnaModel",
    "HyenaDnaPreTrainedModel",
    "HyenaDnaForCausalLM",
    "HyenaDnaForPreTraining",
    "HyenaDnaForSequencePrediction",
    "HyenaDnaForTokenPrediction",
]

AutoConfig.register("hyenadna", HyenaDnaConfig)
AutoBackbone.register(HyenaDnaConfig, HyenaDnaModel)
AutoModel.register(HyenaDnaConfig, HyenaDnaModel)
AutoModelForCausalLM.register(HyenaDnaConfig, HyenaDnaForCausalLM)
AutoModelForSequencePrediction.register(HyenaDnaConfig, HyenaDnaForSequencePrediction)
AutoModelForSequenceClassification.register(HyenaDnaConfig, HyenaDnaForSequencePrediction)
AutoModelForTokenPrediction.register(HyenaDnaConfig, HyenaDnaForTokenPrediction)
AutoModelForPreTraining.register(HyenaDnaConfig, HyenaDnaForPreTraining)
AutoTokenizer.register(HyenaDnaConfig, DnaTokenizer)
