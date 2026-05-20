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
from .configuration_framepool import FramepoolConfig
from .modeling_framepool import (
    FramepoolForSequencePrediction,
    FramepoolForSequencePredictorOutput,
    FramepoolModel,
    FramepoolModelOutput,
    FramepoolPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "FramepoolConfig",
    "FramepoolModel",
    "FramepoolModelOutput",
    "FramepoolPreTrainedModel",
    "FramepoolForSequencePrediction",
    "FramepoolForSequencePredictorOutput",
]

AutoConfig.register("framepool", FramepoolConfig)
AutoModel.register(FramepoolConfig, FramepoolModel)
AutoModelForSequencePrediction.register(FramepoolConfig, FramepoolForSequencePrediction)
AutoModelForMeanRibosomeLoadPrediction.register(FramepoolConfig, FramepoolForSequencePrediction)
AutoModelForSequenceClassification.register(FramepoolConfig, FramepoolForSequencePrediction)
AutoTokenizer.register(FramepoolConfig, RnaTokenizer)
