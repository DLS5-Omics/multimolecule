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
from .configuration_bpnet import BPNetConfig
from .modeling_bpnet import (
    BPNetEncoderOutput,
    BPNetForProfilePrediction,
    BPNetForTokenPrediction,
    BPNetHeadOutput,
    BPNetModel,
    BPNetModelOutput,
    BPNetPreTrainedModel,
    BPNetProfilePredictorOutput,
)

__all__ = [
    "DnaTokenizer",
    "BPNetConfig",
    "BPNetModel",
    "BPNetForProfilePrediction",
    "BPNetForTokenPrediction",
    "BPNetModelOutput",
    "BPNetEncoderOutput",
    "BPNetHeadOutput",
    "BPNetProfilePredictorOutput",
    "BPNetPreTrainedModel",
]

AutoConfig.register("bpnet", BPNetConfig)
AutoModel.register(BPNetConfig, BPNetModel)
AutoModelForTokenPrediction.register(BPNetConfig, BPNetForTokenPrediction)
AutoModelForProfilePrediction.register(BPNetConfig, BPNetForProfilePrediction)
AutoModelForRegulatoryProfilePrediction.register(BPNetConfig, BPNetForProfilePrediction)
AutoModelForTokenClassification.register(BPNetConfig, BPNetForTokenPrediction)
AutoTokenizer.register(BPNetConfig, DnaTokenizer)
