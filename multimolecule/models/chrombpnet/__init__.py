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


from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers import DnaTokenizer

from ..modeling_auto import (
    AutoModelForProfilePrediction,
    AutoModelForRegulatoryProfilePrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    AutoModelForTokenPrediction,
)
from .configuration_chrombpnet import ChromBpNetConfig
from .modeling_chrombpnet import (
    ChromBpNetForProfilePrediction,
    ChromBpNetForTokenPrediction,
    ChromBpNetModel,
    ChromBpNetModelOutput,
    ChromBpNetPreTrainedModel,
    ChromBpNetProfilePredictorOutput,
)

__all__ = [
    "DnaTokenizer",
    "ChromBpNetConfig",
    "ChromBpNetModel",
    "ChromBpNetForProfilePrediction",
    "ChromBpNetForTokenPrediction",
    "ChromBpNetModelOutput",
    "ChromBpNetProfilePredictorOutput",
    "ChromBpNetPreTrainedModel",
]

AutoConfig.register("chrombpnet", ChromBpNetConfig)
AutoModel.register(ChromBpNetConfig, ChromBpNetModel)
AutoModelForTokenPrediction.register(ChromBpNetConfig, ChromBpNetForTokenPrediction)
AutoModelForProfilePrediction.register(ChromBpNetConfig, ChromBpNetForProfilePrediction)
AutoModelForRegulatoryProfilePrediction.register(ChromBpNetConfig, ChromBpNetForProfilePrediction)
AutoModelForRegulatoryVariantEffectPrediction.register(ChromBpNetConfig, ChromBpNetForProfilePrediction)
AutoModelForTokenClassification.register(ChromBpNetConfig, ChromBpNetForTokenPrediction)
AutoTokenizer.register(ChromBpNetConfig, DnaTokenizer)
