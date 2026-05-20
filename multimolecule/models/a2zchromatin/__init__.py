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


from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from multimolecule.tokenisers import DnaTokenizer

from ..modeling_auto import (
    AutoModelForMethylationPrediction,
    AutoModelForRegulatoryActivityPrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    AutoModelForSequencePrediction,
)
from .configuration_a2zchromatin import A2zChromatinConfig
from .modeling_a2zchromatin import (
    A2zChromatinForSequencePrediction,
    A2zChromatinModel,
    A2zChromatinModelOutput,
    A2zChromatinPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "A2zChromatinConfig",
    "A2zChromatinModel",
    "A2zChromatinModelOutput",
    "A2zChromatinPreTrainedModel",
    "A2zChromatinForSequencePrediction",
]

AutoConfig.register("a2zchromatin", A2zChromatinConfig)
AutoModel.register(A2zChromatinConfig, A2zChromatinModel)
AutoModelForSequencePrediction.register(A2zChromatinConfig, A2zChromatinForSequencePrediction)
AutoModelForMethylationPrediction.register(A2zChromatinConfig, A2zChromatinForSequencePrediction)
AutoModelForRegulatoryActivityPrediction.register(A2zChromatinConfig, A2zChromatinForSequencePrediction)
AutoModelForRegulatoryVariantEffectPrediction.register(A2zChromatinConfig, A2zChromatinForSequencePrediction)
AutoModelForSequenceClassification.register(A2zChromatinConfig, A2zChromatinForSequencePrediction)
AutoTokenizer.register(A2zChromatinConfig, DnaTokenizer)
