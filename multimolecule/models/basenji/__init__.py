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

from multimolecule.tokenisers import DnaTokenizer

from ..modeling_auto import (
    AutoModelForRegulatoryTrackPrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    AutoModelForTokenPrediction,
)
from .configuration_basenji import BasenjiBlockConfig, BasenjiConfig
from .modeling_basenji import (
    BasenjiForTokenPrediction,
    BasenjiModel,
    BasenjiPreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "BasenjiConfig",
    "BasenjiBlockConfig",
    "BasenjiModel",
    "BasenjiForTokenPrediction",
    "BasenjiPreTrainedModel",
]

AutoConfig.register("basenji", BasenjiConfig)
AutoModel.register(BasenjiConfig, BasenjiModel)
AutoModelForTokenPrediction.register(BasenjiConfig, BasenjiForTokenPrediction)
AutoModelForRegulatoryTrackPrediction.register(BasenjiConfig, BasenjiForTokenPrediction)
AutoModelForRegulatoryVariantEffectPrediction.register(BasenjiConfig, BasenjiForTokenPrediction)
AutoModelForTokenClassification.register(BasenjiConfig, BasenjiForTokenPrediction)
AutoTokenizer.register(BasenjiConfig, DnaTokenizer)
