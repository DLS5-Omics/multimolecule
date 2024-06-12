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
from .configuration_rnaernie import RnaErnieConfig
from .modeling_rnaernie import (
    RnaErnieForContactPrediction,
    RnaErnieForMaskedLM,
    RnaErnieForNucleotidePrediction,
    RnaErnieForPreTraining,
    RnaErnieForSequencePrediction,
    RnaErnieForTokenPrediction,
    RnaErnieModel,
    RnaErniePreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaErnieConfig",
    "RnaErnieModel",
    "RnaErniePreTrainedModel",
    "RnaErnieForContactPrediction",
    "RnaErnieForNucleotidePrediction",
    "RnaErnieForSequencePrediction",
    "RnaErnieForTokenPrediction",
    "RnaErnieForMaskedLM",
    "RnaErnieForPreTraining",
]

AutoConfig.register("rnaernie", RnaErnieConfig)
AutoBackbone.register(RnaErnieConfig, RnaErnieModel)
AutoModel.register(RnaErnieConfig, RnaErnieModel)
AutoModelForContactPrediction.register(RnaErnieConfig, RnaErnieForContactPrediction)
AutoModelForNucleotidePrediction.register(RnaErnieConfig, RnaErnieForNucleotidePrediction)
AutoModelForSequencePrediction.register(RnaErnieConfig, RnaErnieForSequencePrediction)
AutoModelForSequenceClassification.register(RnaErnieConfig, RnaErnieForSequencePrediction)
AutoModelForTokenPrediction.register(RnaErnieConfig, RnaErnieForTokenPrediction)
AutoModelForTokenClassification.register(RnaErnieConfig, RnaErnieForTokenPrediction)
AutoModelForMaskedLM.register(RnaErnieConfig, RnaErnieForMaskedLM)
AutoModelForPreTraining.register(RnaErnieConfig, RnaErnieForPreTraining)
AutoTokenizer.register(RnaErnieConfig, RnaTokenizer)
