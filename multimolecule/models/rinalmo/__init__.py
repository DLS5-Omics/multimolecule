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
from .configuration_rinalmo import RiNALMoConfig
from .modeling_rinalmo import (
    RiNALMoForMaskedLM,
    RiNALMoForNucleotidePrediction,
    RiNALMoForPreTraining,
    RiNALMoForSequencePrediction,
    RiNALMoForTokenPrediction,
    RiNALMoModel,
    RiNALMoPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoPreTrainedModel",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForNucleotidePrediction",
]

AutoConfig.register("rinalmo", RiNALMoConfig)
AutoBackbone.register(RiNALMoConfig, RiNALMoModel)
AutoModel.register(RiNALMoConfig, RiNALMoModel)
AutoModelForMaskedLM.register(RiNALMoConfig, RiNALMoForMaskedLM)
AutoModelForPreTraining.register(RiNALMoConfig, RiNALMoForPreTraining)
AutoModelForSequenceClassification.register(RiNALMoConfig, RiNALMoForSequencePrediction)
AutoModelForTokenClassification.register(RiNALMoConfig, RiNALMoForTokenPrediction)
AutoModelForNucleotideClassification.register(RiNALMoConfig, RiNALMoForNucleotidePrediction)
AutoTokenizer.register(RiNALMoConfig, RnaTokenizer)
