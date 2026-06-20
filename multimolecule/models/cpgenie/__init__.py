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

from ..modeling_auto import AutoModelForMethylationPrediction, AutoModelForSequencePrediction
from .configuration_cpgenie import CpGenieConfig
from .modeling_cpgenie import (
    CpGenieForSequencePrediction,
    CpGenieModel,
    CpGeniePreTrainedModel,
)

__all__ = [
    "DnaTokenizer",
    "CpGenieConfig",
    "CpGenieModel",
    "CpGeniePreTrainedModel",
    "CpGenieForSequencePrediction",
]

AutoConfig.register("cpgenie", CpGenieConfig)
AutoModel.register(CpGenieConfig, CpGenieModel)
AutoModelForMethylationPrediction.register(CpGenieConfig, CpGenieForSequencePrediction)
AutoModelForSequencePrediction.register(CpGenieConfig, CpGenieForSequencePrediction)
AutoModelForSequenceClassification.register(CpGenieConfig, CpGenieForSequencePrediction)
AutoTokenizer.register(CpGenieConfig, DnaTokenizer)
