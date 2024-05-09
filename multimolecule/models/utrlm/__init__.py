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
from .configuration_utrlm import UtrLmConfig
from .modeling_utrlm import (
    UtrLmForMaskedLM,
    UtrLmForNucleotideClassification,
    UtrLmForPretraining,
    UtrLmForSequenceClassification,
    UtrLmForTokenClassification,
    UtrLmModel,
    UtrLmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmPreTrainedModel",
    "RnaTokenizer",
    "UtrLmForMaskedLM",
    "UtrLmForSequenceClassification",
    "UtrLmForTokenClassification",
    "UtrLmForNucleotideClassification",
]

AutoConfig.register("utrlm", UtrLmConfig)
AutoModel.register(UtrLmConfig, UtrLmModel)
AutoModelForMaskedLM.register(UtrLmConfig, UtrLmForMaskedLM)
AutoModelForPreTraining.register(UtrLmConfig, UtrLmForPretraining)
AutoModelForSequenceClassification.register(UtrLmConfig, UtrLmForSequenceClassification)
AutoModelForTokenClassification.register(UtrLmConfig, UtrLmForTokenClassification)
AutoModelForNucleotideClassification.register(UtrLmConfig, UtrLmForNucleotideClassification)
AutoTokenizer.register(UtrLmConfig, RnaTokenizer)
