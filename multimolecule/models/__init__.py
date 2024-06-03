# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ..tokenisers.rna import RnaTokenizer
from .calm import (
    CaLmConfig,
    CaLmForMaskedLM,
    CaLmForNucleotidePrediction,
    CaLmForPreTraining,
    CaLmForSequencePrediction,
    CaLmForTokenPrediction,
    CaLmModel,
)
from .configuration_utils import BaseHeadConfig, HeadConfig, MaskedLMHeadConfig, PreTrainedConfig
from .ernierna import (
    ErnieRnaConfig,
    ErnieRnaForContactClassification,
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
    RiNALMoForMaskedLM,
    RiNALMoForNucleotidePrediction,
    RiNALMoForPreTraining,
    RiNALMoForSequencePrediction,
    RiNALMoForTokenPrediction,
    RiNALMoModel,
)
from .rnabert import (
    RnaBertConfig,
    RnaBertForMaskedLM,
    RnaBertForNucleotidePrediction,
    RnaBertForPreTraining,
    RnaBertForSequencePrediction,
    RnaBertForTokenPrediction,
    RnaBertModel,
)
from .rnafm import (
    RnaFmConfig,
    RnaFmForMaskedLM,
    RnaFmForNucleotidePrediction,
    RnaFmForPreTraining,
    RnaFmForSequencePrediction,
    RnaFmForTokenPrediction,
    RnaFmModel,
)
from .rnamsm import (
    RnaMsmConfig,
    RnaMsmForMaskedLM,
    RnaMsmForNucleotidePrediction,
    RnaMsmForPreTraining,
    RnaMsmForSequencePrediction,
    RnaMsmForTokenPrediction,
    RnaMsmModel,
)
from .splicebert import (
    SpliceBertConfig,
    SpliceBertForMaskedLM,
    SpliceBertForNucleotidePrediction,
    SpliceBertForPreTraining,
    SpliceBertForSequencePrediction,
    SpliceBertForTokenPrediction,
    SpliceBertModel,
)
from .utrbert import (
    UtrBertConfig,
    UtrBertForMaskedLM,
    UtrBertForNucleotidePrediction,
    UtrBertForPreTraining,
    UtrBertForSequencePrediction,
    UtrBertForTokenPrediction,
    UtrBertModel,
)
from .utrlm import (
    UtrLmConfig,
    UtrLmForMaskedLM,
    UtrLmForNucleotidePrediction,
    UtrLmForPreTraining,
    UtrLmForSequencePrediction,
    UtrLmForTokenPrediction,
    UtrLmModel,
)

__all__ = [
    "PreTrainedConfig",
    "BaseHeadConfig",
    "HeadConfig",
    "MaskedLMHeadConfig",
    "RnaTokenizer",
    "AutoModelForContactPrediction",
    "AutoModelForNucleotidePrediction",
    "AutoModelForSequencePrediction",
    "AutoModelForTokenPrediction",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "CaLmForNucleotidePrediction",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "ErnieRnaConfig",
    "ErnieRnaModel",
    "ErnieRnaForMaskedLM",
    "ErnieRnaForPreTraining",
    "ErnieRnaForNucleotidePrediction",
    "ErnieRnaForSequencePrediction",
    "ErnieRnaForTokenPrediction",
    "ErnieRnaForContactClassification",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RiNALMoForNucleotidePrediction",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaBertForNucleotidePrediction",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaFmForNucleotidePrediction",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "RnaMsmForNucleotidePrediction",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "SpliceBertForNucleotidePrediction",
    "SpliceBertForSequencePrediction",
    "SpliceBertForTokenPrediction",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrBertForNucleotidePrediction",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "UtrLmForNucleotidePrediction",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
]
