# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers.rna import RnaTokenizer

from ..modeling_auto import AutoModelForNucleotideClassification
from .configuration_utrbert import UtrBertConfig
from .modeling_utrbert import (
    UtrBertForMaskedLM,
    UtrBertForNucleotideClassification,
    UtrBertForPreTraining,
    UtrBertForSequenceClassification,
    UtrBertForTokenClassification,
    UtrBertModel,
    UtrBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertPreTrainedModel",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrBertForSequenceClassification",
    "UtrBertForTokenClassification",
    "UtrBertForNucleotideClassification",
]

AutoConfig.register("utrbert", UtrBertConfig)
AutoModel.register(UtrBertConfig, UtrBertModel)
AutoModelForMaskedLM.register(UtrBertConfig, UtrBertForMaskedLM)
AutoModelForPreTraining.register(UtrBertConfig, UtrBertForPreTraining)
AutoModelForSequenceClassification.register(UtrBertConfig, UtrBertForSequenceClassification)
AutoModelForTokenClassification.register(UtrBertConfig, UtrBertForTokenClassification)
AutoModelForNucleotideClassification.register(UtrBertConfig, UtrBertForNucleotideClassification)
AutoTokenizer.register(UtrBertConfig, RnaTokenizer)
