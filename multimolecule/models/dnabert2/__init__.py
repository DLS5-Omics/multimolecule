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
    PreTrainedTokenizerFast,
)

from ..modeling_auto import AutoModelForContactPrediction, AutoModelForSequencePrediction, AutoModelForTokenPrediction
from .configuration_dnabert2 import DnaBert2Config
from .modeling_dnabert2 import (
    DnaBert2ForContactPrediction,
    DnaBert2ForMaskedLM,
    DnaBert2ForPreTraining,
    DnaBert2ForSequencePrediction,
    DnaBert2ForTokenPrediction,
    DnaBert2Model,
    DnaBert2PreTrainedModel,
)

__all__ = [
    "DnaBert2Config",
    "DnaBert2Model",
    "DnaBert2PreTrainedModel",
    "DnaBert2ForContactPrediction",
    "DnaBert2ForSequencePrediction",
    "DnaBert2ForTokenPrediction",
    "DnaBert2ForMaskedLM",
    "DnaBert2ForPreTraining",
]

AutoConfig.register("dnabert2", DnaBert2Config)
AutoBackbone.register(DnaBert2Config, DnaBert2Model)
AutoModel.register(DnaBert2Config, DnaBert2Model)
AutoModelForContactPrediction.register(DnaBert2Config, DnaBert2ForContactPrediction)
AutoModelForSequencePrediction.register(DnaBert2Config, DnaBert2ForSequencePrediction)
AutoModelForSequenceClassification.register(DnaBert2Config, DnaBert2ForSequencePrediction)
AutoModelForTokenPrediction.register(DnaBert2Config, DnaBert2ForTokenPrediction)
AutoModelForTokenClassification.register(DnaBert2Config, DnaBert2ForTokenPrediction)
AutoModelForMaskedLM.register(DnaBert2Config, DnaBert2ForMaskedLM)
AutoModelForPreTraining.register(DnaBert2Config, DnaBert2ForPreTraining)
AutoTokenizer.register(DnaBert2Config, slow_tokenizer_class=None, fast_tokenizer_class=PreTrainedTokenizerFast)
