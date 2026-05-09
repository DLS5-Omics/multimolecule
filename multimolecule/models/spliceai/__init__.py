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

from ..modeling_auto import AutoModelForTokenPrediction
from .configuration_spliceai import SpliceAiConfig, SpliceAiStageConfig
from .modeling_spliceai import (
    SpliceAiForTokenPrediction,
    SpliceAiModel,
    SpliceAiModelOutput,
    SpliceAiModuleOutput,
    SpliceAiPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "SpliceAiConfig",
    "SpliceAiStageConfig",
    "SpliceAiModel",
    "SpliceAiForTokenPrediction",
    "SpliceAiModelOutput",
    "SpliceAiModuleOutput",
    "SpliceAiPreTrainedModel",
]

AutoConfig.register("spliceai", SpliceAiConfig)
AutoModel.register(SpliceAiConfig, SpliceAiModel)
AutoModelForTokenPrediction.register(SpliceAiConfig, SpliceAiForTokenPrediction)
AutoModelForTokenClassification.register(SpliceAiConfig, SpliceAiForTokenPrediction)
AutoTokenizer.register(SpliceAiConfig, RnaTokenizer)
