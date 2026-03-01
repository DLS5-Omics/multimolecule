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
from .configuration_dnaberts import DnaBertSConfig
from .modeling_dnaberts import (
    DnaBertSForContactPrediction,
    DnaBertSForMaskedLM,
    DnaBertSForPreTraining,
    DnaBertSForSequencePrediction,
    DnaBertSForTokenPrediction,
    DnaBertSModel,
    DnaBertSPreTrainedModel,
)

__all__ = [
    "DnaBertSConfig",
    "DnaBertSModel",
    "DnaBertSPreTrainedModel",
    "DnaBertSForContactPrediction",
    "DnaBertSForSequencePrediction",
    "DnaBertSForTokenPrediction",
    "DnaBertSForMaskedLM",
    "DnaBertSForPreTraining",
]

AutoConfig.register("dnaberts", DnaBertSConfig)
AutoBackbone.register(DnaBertSConfig, DnaBertSModel)
AutoModel.register(DnaBertSConfig, DnaBertSModel)
AutoModelForContactPrediction.register(DnaBertSConfig, DnaBertSForContactPrediction)
AutoModelForSequencePrediction.register(DnaBertSConfig, DnaBertSForSequencePrediction)
AutoModelForSequenceClassification.register(DnaBertSConfig, DnaBertSForSequencePrediction)
AutoModelForTokenPrediction.register(DnaBertSConfig, DnaBertSForTokenPrediction)
AutoModelForTokenClassification.register(DnaBertSConfig, DnaBertSForTokenPrediction)
AutoModelForMaskedLM.register(DnaBertSConfig, DnaBertSForMaskedLM)
AutoModelForPreTraining.register(DnaBertSConfig, DnaBertSForPreTraining)
AutoTokenizer.register(DnaBertSConfig, slow_tokenizer_class=None, fast_tokenizer_class=PreTrainedTokenizerFast)
