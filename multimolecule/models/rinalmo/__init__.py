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

from ..modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForNucleotidePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .configuration_rinalmo import RiNALMoConfig
from .modeling_rinalmo import (
    RiNALMoForContactPrediction,
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
    "RiNALMoForContactPrediction",
    "RiNALMoForNucleotidePrediction",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
]

AutoConfig.register("rinalmo", RiNALMoConfig)
AutoBackbone.register(RiNALMoConfig, RiNALMoModel)
AutoModel.register(RiNALMoConfig, RiNALMoModel)
AutoModelForContactPrediction.register(RiNALMoConfig, RiNALMoForContactPrediction)
AutoModelForNucleotidePrediction.register(RiNALMoConfig, RiNALMoForNucleotidePrediction)
AutoModelForSequencePrediction.register(RiNALMoConfig, RiNALMoForSequencePrediction)
AutoModelForSequenceClassification.register(RiNALMoConfig, RiNALMoForSequencePrediction)
AutoModelForTokenPrediction.register(RiNALMoConfig, RiNALMoForTokenPrediction)
AutoModelForTokenClassification.register(RiNALMoConfig, RiNALMoForTokenPrediction)
AutoModelForMaskedLM.register(RiNALMoConfig, RiNALMoForMaskedLM)
AutoModelForPreTraining.register(RiNALMoConfig, RiNALMoForPreTraining)
AutoTokenizer.register(RiNALMoConfig, RnaTokenizer)
