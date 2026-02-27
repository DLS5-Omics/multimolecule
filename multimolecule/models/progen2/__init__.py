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

from multimolecule.tokenisers import ProteinTokenizer

from ..modeling_auto import AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_progen2 import ProGen2Config
from .modeling_progen2 import (
    ProGen2ForCausalLM,
    ProGen2ForPreTraining,
    ProGen2ForSequencePrediction,
    ProGen2ForTokenPrediction,
    ProGen2Model,
    ProGen2PreTrainedModel,
)

__all__ = [
    "ProteinTokenizer",
    "ProGen2Config",
    "ProGen2Model",
    "ProGen2PreTrainedModel",
    "ProGen2ForCausalLM",
    "ProGen2ForPreTraining",
    "ProGen2ForSequencePrediction",
    "ProGen2ForTokenPrediction",
]

AutoConfig.register("progen2", ProGen2Config)
AutoBackbone.register(ProGen2Config, ProGen2Model)
AutoModel.register(ProGen2Config, ProGen2Model)
AutoModelForCausalLM.register(ProGen2Config, ProGen2ForCausalLM)
AutoModelForSequencePrediction.register(ProGen2Config, ProGen2ForSequencePrediction)
AutoModelForSequenceClassification.register(ProGen2Config, ProGen2ForSequencePrediction)
AutoModelForTokenPrediction.register(ProGen2Config, ProGen2ForTokenPrediction)
AutoModelForPreTraining.register(ProGen2Config, ProGen2ForPreTraining)
AutoTokenizer.register(ProGen2Config, ProteinTokenizer)
