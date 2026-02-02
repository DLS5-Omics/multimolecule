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


from multimolecule.modules import HeadConfig
from multimolecule.tokenisers import DnaTokenizer, ProteinTokenizer, RnaTokenizer

from .calm import (
    CaLmConfig,
    CaLmForContactPrediction,
    CaLmForMaskedLM,
    CaLmForPreTraining,
    CaLmForSequencePrediction,
    CaLmForTokenPrediction,
    CaLmModel,
)
from .configuration_utils import PreTrainedConfig
from .ernierna import (
    ErnieRnaConfig,
    ErnieRnaForContactPrediction,
    ErnieRnaForMaskedLM,
    ErnieRnaForPreTraining,
    ErnieRnaForSecondaryStructurePrediction,
    ErnieRnaForSequencePrediction,
    ErnieRnaForTokenPrediction,
    ErnieRnaModel,
)
from .modeling_auto import (
    AutoModelForContactPrediction,
    AutoModelForRnaSecondaryStructurePrediction,
    AutoModelForSequencePrediction,
    AutoModelForTokenPrediction,
)
from .ribonanzanet import (
    RibonanzaNetConfig,
    RibonanzaNetForContactPrediction,
    RibonanzaNetForDegradationPrediction,
    RibonanzaNetForPreTraining,
    RibonanzaNetForSecondaryStructurePrediction,
    RibonanzaNetForSequenceDropoutPrediction,
    RibonanzaNetForSequencePrediction,
    RibonanzaNetForTokenPrediction,
    RibonanzaNetModel,
)
from .rinalmo import (
    RiNALMoConfig,
    RiNALMoForContactPrediction,
    RiNALMoForMaskedLM,
    RiNALMoForPreTraining,
    RiNALMoForSecondaryStructurePrediction,
    RiNALMoForSequencePrediction,
    RiNALMoForTokenPrediction,
    RiNALMoModel,
)
from .rnabert import (
    RnaBertConfig,
    RnaBertForContactPrediction,
    RnaBertForMaskedLM,
    RnaBertForPreTraining,
    RnaBertForSequencePrediction,
    RnaBertForTokenPrediction,
    RnaBertModel,
)
from .rnaernie import (
    RnaErnieConfig,
    RnaErnieForContactPrediction,
    RnaErnieForMaskedLM,
    RnaErnieForPreTraining,
    RnaErnieForSequencePrediction,
    RnaErnieForTokenPrediction,
    RnaErnieModel,
)
from .rnafm import (
    RnaFmConfig,
    RnaFmForContactPrediction,
    RnaFmForMaskedLM,
    RnaFmForPreTraining,
    RnaFmForSecondaryStructurePrediction,
    RnaFmForSequencePrediction,
    RnaFmForTokenPrediction,
    RnaFmModel,
)
from .rnamsm import (
    RnaMsmConfig,
    RnaMsmForContactPrediction,
    RnaMsmForMaskedLM,
    RnaMsmForPreTraining,
    RnaMsmForSecondaryStructurePrediction,
    RnaMsmForSequencePrediction,
    RnaMsmForTokenPrediction,
    RnaMsmModel,
)
from .spliceai import (
    SpliceAiConfig,
    SpliceAiModel,
)
from .splicebert import (
    SpliceBertConfig,
    SpliceBertForContactPrediction,
    SpliceBertForMaskedLM,
    SpliceBertForPreTraining,
    SpliceBertForSequencePrediction,
    SpliceBertForTokenPrediction,
    SpliceBertModel,
)
from .utrbert import (
    UtrBertConfig,
    UtrBertForContactPrediction,
    UtrBertForMaskedLM,
    UtrBertForPreTraining,
    UtrBertForSequencePrediction,
    UtrBertForTokenPrediction,
    UtrBertModel,
)
from .utrlm import (
    UtrLmConfig,
    UtrLmForContactPrediction,
    UtrLmForMaskedLM,
    UtrLmForPreTraining,
    UtrLmForSecondaryStructurePrediction,
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
    "AutoModelForSequencePrediction",
    "AutoModelForTokenPrediction",
    "AutoModelForRnaSecondaryStructurePrediction",
    "CaLmConfig",
    "CaLmModel",
    "CaLmForContactPrediction",
    "CaLmForSequencePrediction",
    "CaLmForTokenPrediction",
    "CaLmForMaskedLM",
    "CaLmForPreTraining",
    "ErnieRnaConfig",
    "ErnieRnaModel",
    "ErnieRnaForContactPrediction",
    "ErnieRnaForSequencePrediction",
    "ErnieRnaForTokenPrediction",
    "ErnieRnaForMaskedLM",
    "ErnieRnaForPreTraining",
    "ErnieRnaForSecondaryStructurePrediction",
    "RibonanzaNetConfig",
    "RibonanzaNetModel",
    "RibonanzaNetForContactPrediction",
    "RibonanzaNetForSequencePrediction",
    "RibonanzaNetForTokenPrediction",
    "RibonanzaNetForPreTraining",
    "RibonanzaNetForSecondaryStructurePrediction",
    "RibonanzaNetForDegradationPrediction",
    "RibonanzaNetForSequenceDropoutPrediction",
    "RiNALMoConfig",
    "RiNALMoModel",
    "RiNALMoForContactPrediction",
    "RiNALMoForSequencePrediction",
    "RiNALMoForTokenPrediction",
    "RiNALMoForMaskedLM",
    "RiNALMoForPreTraining",
    "RiNALMoForSecondaryStructurePrediction",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertForContactPrediction",
    "RnaBertForSequencePrediction",
    "RnaBertForTokenPrediction",
    "RnaBertForMaskedLM",
    "RnaBertForPreTraining",
    "RnaErnieConfig",
    "RnaErnieModel",
    "RnaErnieForContactPrediction",
    "RnaErnieForSequencePrediction",
    "RnaErnieForTokenPrediction",
    "RnaErnieForMaskedLM",
    "RnaErnieForPreTraining",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmForContactPrediction",
    "RnaFmForSequencePrediction",
    "RnaFmForTokenPrediction",
    "RnaFmForMaskedLM",
    "RnaFmForPreTraining",
    "RnaFmForSecondaryStructurePrediction",
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaMsmForContactPrediction",
    "RnaMsmForSequencePrediction",
    "RnaMsmForTokenPrediction",
    "RnaMsmForMaskedLM",
    "RnaMsmForPreTraining",
    "RnaMsmForSecondaryStructurePrediction",
    "SpliceAiConfig",
    "SpliceAiModel",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForContactPrediction",
    "SpliceBertForSequencePrediction",
    "SpliceBertForTokenPrediction",
    "SpliceBertForMaskedLM",
    "SpliceBertForPreTraining",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertForContactPrediction",
    "UtrBertForSequencePrediction",
    "UtrBertForTokenPrediction",
    "UtrBertForMaskedLM",
    "UtrBertForPreTraining",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmForContactPrediction",
    "UtrLmForSequencePrediction",
    "UtrLmForTokenPrediction",
    "UtrLmForMaskedLM",
    "UtrLmForPreTraining",
    "UtrLmForSecondaryStructurePrediction",
]
