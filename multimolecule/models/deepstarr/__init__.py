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
    AutoModelForRegulatoryActivityPrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    AutoModelForSequencePrediction,
)
from .configuration_deepstarr import DeepStarrConfig
from .modeling_deepstarr import (
    DeepStarrForSequencePrediction,
    DeepStarrModel,
    DeepStarrModelOutput,
    DeepStarrPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "DeepStarrConfig",
    "DeepStarrModel",
    "DeepStarrForSequencePrediction",
    "DeepStarrModelOutput",
    "DeepStarrPreTrainedModel",
]

AutoConfig.register("deepstarr", DeepStarrConfig)
AutoModel.register(DeepStarrConfig, DeepStarrModel)
AutoModelForSequencePrediction.register(DeepStarrConfig, DeepStarrForSequencePrediction)
AutoModelForRegulatoryActivityPrediction.register(DeepStarrConfig, DeepStarrForSequencePrediction)
AutoModelForRegulatoryVariantEffectPrediction.register(DeepStarrConfig, DeepStarrForSequencePrediction)
AutoModelForSequenceClassification.register(DeepStarrConfig, DeepStarrForSequencePrediction)
AutoTokenizer.register(DeepStarrConfig, DnaTokenizer)
