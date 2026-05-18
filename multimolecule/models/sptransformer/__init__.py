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


from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer

from multimolecule.tokenisers import RnaTokenizer

from ..modeling_auto import (
    AutoModelForSpliceSitePrediction,
    AutoModelForSpliceVariantEffectPrediction,
    AutoModelForTokenPrediction,
)
from .configuration_sptransformer import SpTransformerConfig, SpTransformerFeatureEncoderConfig
from .modeling_sptransformer import (
    SpTransformerAttentionMap,
    SpTransformerForTokenPrediction,
    SpTransformerModel,
    SpTransformerModelOutput,
    SpTransformerPreTrainedModel,
    SpTransformerTokenPredictorOutput,
)

__all__ = [
    "RnaTokenizer",
    "SpTransformerConfig",
    "SpTransformerFeatureEncoderConfig",
    "SpTransformerModel",
    "SpTransformerForTokenPrediction",
    "SpTransformerModelOutput",
    "SpTransformerAttentionMap",
    "SpTransformerPreTrainedModel",
    "SpTransformerTokenPredictorOutput",
]

AutoConfig.register("sptransformer", SpTransformerConfig)
AutoModel.register(SpTransformerConfig, SpTransformerModel)
AutoModelForTokenPrediction.register(SpTransformerConfig, SpTransformerForTokenPrediction)
AutoModelForSpliceSitePrediction.register(SpTransformerConfig, SpTransformerModel)
AutoModelForSpliceVariantEffectPrediction.register(SpTransformerConfig, SpTransformerModel)
AutoModelForTokenClassification.register(SpTransformerConfig, SpTransformerForTokenPrediction)
AutoTokenizer.register(SpTransformerConfig, RnaTokenizer)
