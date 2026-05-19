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
    AutoModelForRegulatorySequencePrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    AutoModelForSequencePrediction,
)
from .configuration_aparent import AparentConfig
from .modeling_aparent import (
    AparentForSequencePrediction,
    AparentModel,
    AparentModelOutput,
    AparentPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "AparentConfig",
    "AparentModel",
    "AparentPreTrainedModel",
    "AparentForSequencePrediction",
    "AparentModelOutput",
]

AutoConfig.register("aparent", AparentConfig)
AutoModel.register(AparentConfig, AparentModel)
AutoModelForSequencePrediction.register(AparentConfig, AparentForSequencePrediction)
AutoModelForRegulatorySequencePrediction.register(AparentConfig, AparentForSequencePrediction)
AutoModelForRegulatoryVariantEffectPrediction.register(AparentConfig, AparentForSequencePrediction)
AutoModelForSequenceClassification.register(AparentConfig, AparentForSequencePrediction)
AutoTokenizer.register(AparentConfig, DnaTokenizer)
