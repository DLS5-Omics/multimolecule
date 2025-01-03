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


from multimolecule.module import HeadConfig
from multimolecule.tokenisers import DnaTokenizer, ProteinTokenizer, RnaTokenizer

from .calm import (
    CaLmConfig,
    CaLmForContactPrediction,
    CaLmForMaskedLM,
    CaLmForNucleotidePrediction,
    CaLmForPreTraining,
    CaLmForSequencePrediction,
    CaLmForTokenPrediction,
    CaLmModel,
)
from .configuration_utils import PreTrainedConfig
from .ernierna import (
    ErnieRnaConfig,
    ErnieRnaForContactClassification,
    ErnieRnaForContactPrediction,
    ErnieRnaForMaskedLM,
    ErnieRnaForNucleotidePrediction,
    ErnieRnaForPreTraining,
    ErnieRnaForSequencePrediction,
    ErnieRnaForTokenPrediction,
    ErnieRnaModel,
)
from .modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForNucleotidePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .rinalmo import (
    RiNALMoConfig,
    RiNALMoForContactPrediction,
    RiNALMoForMaskedLM,
    RiNALMoForNucleotidePrediction,
    RiNALMoForPreTraining,
    RiNALMoForSequencePrediction,
    RiNALMoForTokenPrediction,
    RiNALMoModel,
)
from .rnabert import (
    RnaBertConfig,
    RnaBertForContactPrediction,
    RnaBertForMaskedLM,
    RnaBertForNucleotidePrediction,
    RnaBertForPreTraining,
    RnaBertForSequencePrediction,
    RnaBertForTokenPrediction,
    RnaBertModel,
)
from .rnaernie import (
    RnaErnieConfig,
    RnaErnieForContactPrediction,
    RnaErnieForMaskedLM,
    RnaErnieForNucleotidePrediction,
    RnaErnieForPreTraining,
    RnaErnieForSequencePrediction,
    RnaErnieForTokenPrediction,
    RnaErnieModel,
)
from .rnafm import (
    RnaFmConfig,
    RnaFmForContactPrediction,
    RnaFmForMaskedLM,
    RnaFmForNucleotidePrediction,
    RnaFmForPreTraining,
    RnaFmForSequencePrediction,
    RnaFmForTokenPrediction,
    RnaFmModel,
)
from .rnamsm import (
    RnaMsmConfig,
    RnaMsmForContactPrediction,
    RnaMsmForMaskedLM,
    RnaMsmForNucleotidePrediction,
    RnaMsmForPreTraining,
    RnaMsmForSequencePrediction,
    RnaMsmForTokenPrediction,
    RnaMsmModel,
)
from .splicebert import (
    SpliceBertConfig,
    SpliceBertForContactPrediction,
    SpliceBertForMaskedLM,
    SpliceBertForNucleotidePrediction,
    SpliceBertForPreTraining,
    SpliceBertForSequencePrediction,
    SpliceBertForTokenPrediction,
    SpliceBertModel,
)
from .utrbert import (
    UtrBertConfig,
    UtrBertForContactPrediction,
    UtrBertForMaskedLM,
    UtrBertForNucleotidePrediction,
    UtrBertForPreTraining,
    UtrBertForSequencePrediction,
    UtrBertForTokenPrediction,
    UtrBertModel,
)
from .utrlm import (
    UtrLmConfig,
    UtrLmForContactPrediction,
    UtrLmForMaskedLM,
    UtrLmForNucleotidePrediction,
    UtrLmForPreTraining,
    UtrLmForSequencePrediction,
    UtrLmForTokenPrediction,
    UtrLmModel,
)

__all__ = [
    "PreTrainedConfig",
    "HeadConfig",
    "DnaTokenizer",
    "RnaTokenizer",
    "ProteinTokenizer",
    "AutoModelForContactPrediction",
    "AutoModelForNucleotidePrediction",
    "AutoModelForSequencePrediction",
    "AutoModelForTokenPrediction",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForContactPrediction",
    "CaLmForNucleotidePrediction",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "ErnieRnaConfig",
    "ErnieRnaModel",
    "ErnieRnaForContactPrediction",
    "ErnieRnaForNucleotidePrediction",
    "ErnieRnaForSequencePrediction",
    "ErnieRnaForTokenPrediction",
    "ErnieRnaForMaskedLM",
    "ErnieRnaForPreTraining",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoForContactPrediction",
    "RiNALMoForNucleotidePrediction",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForContactPrediction",
    "RnaBertForNucleotidePrediction",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaErnieConfig",
    "RnaErnieModel",
    "RnaErnieForContactPrediction",
    "RnaErnieForNucleotidePrediction",
    "RnaErnieForSequencePrediction",
    "RnaErnieForTokenPrediction",
    "RnaErnieForMaskedLM",
    "RnaErnieForPreTraining",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForContactPrediction",
    "RnaFmForNucleotidePrediction",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForContactPrediction",
    "RnaMsmForNucleotidePrediction",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForContactPrediction",
    "SpliceBertForNucleotidePrediction",
    "SpliceBertForSequencePrediction",
    "SpliceBertForTokenPrediction",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForContactPrediction",
    "UtrBertForNucleotidePrediction",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForContactPrediction",
    "UtrLmForNucleotidePrediction",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "ErnieRnaForContactClassification",
]
