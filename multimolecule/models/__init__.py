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
from .modeling_auto import AutoModelForNucleotideClassification
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
    "AutoModelForNucleotideClassification",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "CaLmForNucleotidePrediction",
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
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForNucleotidePrediction",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaBertForNucleotidePrediction",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaFmForNucleotidePrediction",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "RnaMsmForNucleotidePrediction",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "SpliceBertForSequencePrediction",
    "SpliceBertForTokenPrediction",
    "SpliceBertForNucleotidePrediction",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrBertForNucleotidePrediction",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
    "UtrLmForNucleotidePrediction",
]
