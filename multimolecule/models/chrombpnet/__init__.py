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
    AutoModelForTokenPrediction,
)
from .configuration_chrombpnet import ChromBPNetConfig
from .modeling_chrombpnet import (
    ChromBPNetBranchOutput,
    ChromBPNetForProfilePrediction,
    ChromBPNetForTokenPrediction,
    ChromBPNetHeadOutput,
    ChromBPNetModel,
    ChromBPNetModelOutput,
    ChromBPNetPreTrainedModel,
    ChromBPNetProfilePredictorOutput,
)

__all__ = [
    "DnaTokenizer",
    "ChromBPNetConfig",
    "ChromBPNetModel",
    "ChromBPNetForProfilePrediction",
    "ChromBPNetForTokenPrediction",
    "ChromBPNetModelOutput",
    "ChromBPNetBranchOutput",
    "ChromBPNetHeadOutput",
    "ChromBPNetProfilePredictorOutput",
    "ChromBPNetPreTrainedModel",
]

AutoConfig.register("chrombpnet", ChromBPNetConfig)
AutoModel.register(ChromBPNetConfig, ChromBPNetModel)
AutoModelForTokenPrediction.register(ChromBPNetConfig, ChromBPNetForTokenPrediction)
AutoModelForProfilePrediction.register(ChromBPNetConfig, ChromBPNetForProfilePrediction)
AutoModelForRegulatoryProfilePrediction.register(ChromBPNetConfig, ChromBPNetForProfilePrediction)
AutoModelForTokenClassification.register(ChromBPNetConfig, ChromBPNetForTokenPrediction)
AutoTokenizer.register(ChromBPNetConfig, DnaTokenizer)
