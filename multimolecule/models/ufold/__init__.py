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


from transformers import AutoConfig, AutoModel, AutoTokenizer

from multimolecule.tokenisers import RnaTokenizer

from ..modeling_auto import AutoModelForRnaSecondaryStructurePrediction
from .configuration_ufold import UfoldConfig
from .modeling_ufold import UfoldModel, UfoldModelOutput, UfoldPreTrainedModel

__all__ = [
    "RnaTokenizer",
    "UfoldConfig",
    "UfoldModel",
    "UfoldModelOutput",
    "UfoldPreTrainedModel",
]

AutoConfig.register("ufold", UfoldConfig)
AutoModel.register(UfoldConfig, UfoldModel)
AutoModelForRnaSecondaryStructurePrediction.register(UfoldConfig, UfoldModel)
AutoTokenizer.register(UfoldConfig, RnaTokenizer)
