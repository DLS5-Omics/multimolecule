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
    AutoBackbone,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers import RnaTokenizer

from ..modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForRnaSecondaryStructurePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .configuration_unirna import UniRnaConfig
from .modeling_unirna import (
    UniRnaForContactPrediction,
    UniRnaForMaskedLM,
    UniRnaForPreTraining,
    UniRnaForSecondaryStructurePrediction,
    UniRnaForSequencePrediction,
    UniRnaForTokenPrediction,
    UniRnaModel,
    UniRnaPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UniRnaConfig",
    "UniRnaModel",
    "UniRnaPreTrainedModel",
    "UniRnaForContactPrediction",
    "UniRnaForSequencePrediction",
    "UniRnaForTokenPrediction",
    "UniRnaForMaskedLM",
    "UniRnaForPreTraining",
    "UniRnaForSecondaryStructurePrediction",
]

AutoConfig.register("unirna", UniRnaConfig)
AutoBackbone.register(UniRnaConfig, UniRnaModel)
AutoModel.register(UniRnaConfig, UniRnaModel)
AutoModelForContactPrediction.register(UniRnaConfig, UniRnaForContactPrediction)
AutoModelForSequencePrediction.register(UniRnaConfig, UniRnaForSequencePrediction)
AutoModelForSequenceClassification.register(UniRnaConfig, UniRnaForSequencePrediction)
AutoModelForTokenPrediction.register(UniRnaConfig, UniRnaForTokenPrediction)
AutoModelForTokenClassification.register(UniRnaConfig, UniRnaForTokenPrediction)
AutoModelForMaskedLM.register(UniRnaConfig, UniRnaForMaskedLM)
AutoModelForPreTraining.register(UniRnaConfig, UniRnaForPreTraining)
AutoModelForRnaSecondaryStructurePrediction.register(UniRnaConfig, UniRnaForSecondaryStructurePrediction)
AutoTokenizer.register(UniRnaConfig, RnaTokenizer)
