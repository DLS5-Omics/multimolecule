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
    AutoBackbone,
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
from .configuration_splicebert import SpliceBertConfig
from .modeling_splicebert import (
    SpliceBertForMaskedLM,
    SpliceBertForNucleotideClassification,
    SpliceBertForPretraining,
    SpliceBertForSequenceClassification,
    SpliceBertForTokenClassification,
    SpliceBertModel,
    SpliceBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertPreTrainedModel",
    "SpliceBertForMaskedLM",
    "SpliceBertForPretraining",
    "SpliceBertForSequenceClassification",
    "SpliceBertForTokenClassification",
    "SpliceBertForNucleotideClassification",
]

AutoConfig.register("splicebert", SpliceBertConfig)
AutoBackbone.register(SpliceBertConfig, SpliceBertModel)
AutoModel.register(SpliceBertConfig, SpliceBertModel)
AutoModelForMaskedLM.register(SpliceBertConfig, SpliceBertForMaskedLM)
AutoModelForPreTraining.register(SpliceBertConfig, SpliceBertForPretraining)
AutoModelForSequenceClassification.register(SpliceBertConfig, SpliceBertForSequenceClassification)
AutoModelForTokenClassification.register(SpliceBertConfig, SpliceBertForTokenClassification)
AutoModelForNucleotideClassification.register(SpliceBertConfig, SpliceBertForNucleotideClassification)
AutoTokenizer.register(SpliceBertConfig, RnaTokenizer)
