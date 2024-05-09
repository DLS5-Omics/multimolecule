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
from .configuration_rnafm import RnaFmConfig
from .modeling_rnafm import (
    RnaFmForMaskedLM,
    RnaFmForNucleotideClassification,
    RnaFmForPretraining,
    RnaFmForSequenceClassification,
    RnaFmForTokenClassification,
    RnaFmModel,
    RnaFmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmPreTrainedModel",
    "RnaFmForMaskedLM",
    "RnaFmForPretraining",
    "RnaFmForSequenceClassification",
    "RnaFmForTokenClassification",
    "RnaFmForNucleotideClassification",
]

AutoConfig.register("rnafm", RnaFmConfig)
AutoModel.register(RnaFmConfig, RnaFmModel)
AutoModelForMaskedLM.register(RnaFmConfig, RnaFmForMaskedLM)
AutoModelForPreTraining.register(RnaFmConfig, RnaFmForPretraining)
AutoModelForSequenceClassification.register(RnaFmConfig, RnaFmForSequenceClassification)
AutoModelForTokenClassification.register(RnaFmConfig, RnaFmForTokenClassification)
AutoModelForNucleotideClassification.register(RnaFmConfig, RnaFmForNucleotideClassification)
AutoTokenizer.register(RnaFmConfig, RnaTokenizer)
