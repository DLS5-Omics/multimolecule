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


from transformers import AutoConfig, AutoModel

from ..modeling_auto import AutoModelForRnaSecondaryStructurePrediction
from .configuration_mxfold2 import Mxfold2Config
from .modeling_mxfold2 import Mxfold2Model, Mxfold2ModelOutput, Mxfold2PreTrainedModel

__all__ = [
    "Mxfold2Config",
    "Mxfold2Model",
    "Mxfold2ModelOutput",
    "Mxfold2PreTrainedModel",
]

AutoConfig.register("mxfold2", Mxfold2Config)
AutoModel.register(Mxfold2Config, Mxfold2Model)
AutoModelForRnaSecondaryStructurePrediction.register(Mxfold2Config, Mxfold2Model)
