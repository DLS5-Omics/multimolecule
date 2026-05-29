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
from .configuration_deltasplice import DeltaSpliceConfig, DeltaSpliceLayerConfig
from .modeling_deltasplice import (
    DeltaSpliceForTokenPrediction,
    DeltaSpliceModel,
    DeltaSpliceModelOutput,
    DeltaSplicePreTrainedModel,
    DeltaSpliceTokenPredictorOutput,
)

__all__ = [
    "RnaTokenizer",
    "DeltaSpliceConfig",
    "DeltaSpliceLayerConfig",
    "DeltaSpliceModel",
    "DeltaSpliceForTokenPrediction",
    "DeltaSpliceModelOutput",
    "DeltaSplicePreTrainedModel",
    "DeltaSpliceTokenPredictorOutput",
]

AutoConfig.register("deltasplice", DeltaSpliceConfig)
AutoModel.register(DeltaSpliceConfig, DeltaSpliceModel)
AutoModelForTokenPrediction.register(DeltaSpliceConfig, DeltaSpliceForTokenPrediction)
# DeltaSplice's native splice-site and variant-effect predictions (including the alternative-sequence delta) are
# produced by the backbone itself via `DeltaSpliceModel.postprocess`; only the backbone implements variant effect.
AutoModelForSpliceSitePrediction.register(DeltaSpliceConfig, DeltaSpliceModel)
AutoModelForSpliceVariantEffectPrediction.register(DeltaSpliceConfig, DeltaSpliceModel)
AutoModelForTokenClassification.register(DeltaSpliceConfig, DeltaSpliceForTokenPrediction)
AutoTokenizer.register(DeltaSpliceConfig, RnaTokenizer)
