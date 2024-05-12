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
from .configuration_rnabert import RnaBertConfig
from .modeling_rnabert import (
    RnaBertForMaskedLM,
    RnaBertForNucleotideClassification,
    RnaBertForPretraining,
    RnaBertForSequenceClassification,
    RnaBertForTokenClassification,
    RnaBertModel,
    RnaBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertPreTrainedModel",
    "RnaBertForMaskedLM",
    "RnaBertForPretraining",
    "RnaBertForSequenceClassification",
    "RnaBertForTokenClassification",
    "RnaBertForNucleotideClassification",
]

AutoConfig.register("rnabert", RnaBertConfig)
AutoModel.register(RnaBertConfig, RnaBertModel)
AutoModelForMaskedLM.register(RnaBertConfig, RnaBertForMaskedLM)
AutoModelForPreTraining.register(RnaBertConfig, RnaBertForPretraining)
AutoModelForSequenceClassification.register(RnaBertConfig, RnaBertForSequenceClassification)
AutoModelForTokenClassification.register(RnaBertConfig, RnaBertForTokenClassification)
AutoModelForNucleotideClassification.register(RnaBertConfig, RnaBertForNucleotideClassification)
AutoTokenizer.register(RnaBertConfig, RnaTokenizer)
