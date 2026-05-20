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

from multimolecule.tokenisers import RnaTokenizer

from ..modeling_auto import AutoModelForMeanRibosomeLoadPrediction, AutoModelForSequencePrediction
from .configuration_optimus5prime import Optimus5PrimeConfig
from .modeling_optimus5prime import (
    Optimus5PrimeForSequencePrediction,
    Optimus5PrimeModel,
    Optimus5PrimeModelOutput,
    Optimus5PrimePreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "Optimus5PrimeConfig",
    "Optimus5PrimeModel",
    "Optimus5PrimePreTrainedModel",
    "Optimus5PrimeForSequencePrediction",
    "Optimus5PrimeModelOutput",
]

AutoConfig.register("optimus5prime", Optimus5PrimeConfig)
AutoModel.register(Optimus5PrimeConfig, Optimus5PrimeModel)
AutoModelForSequencePrediction.register(Optimus5PrimeConfig, Optimus5PrimeForSequencePrediction)
AutoModelForMeanRibosomeLoadPrediction.register(Optimus5PrimeConfig, Optimus5PrimeForSequencePrediction)
AutoModelForSequenceClassification.register(Optimus5PrimeConfig, Optimus5PrimeForSequencePrediction)
AutoTokenizer.register(Optimus5PrimeConfig, RnaTokenizer)
