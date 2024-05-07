from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenisers.rna import RnaTokenizer

from ..modeling_auto import AutoModelForNucleotideClassification
from .configuration_utrlm import UtrLmConfig
from .modeling_utrlm import (
    UtrLmForMaskedLM,
    UtrLmForNucleotideClassification,
    UtrLmForPretraining,
    UtrLmForSequenceClassification,
    UtrLmForTokenClassification,
    UtrLmModel,
    UtrLmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UtrLmConfig",
    "UtrLmModel",
    "UtrLmPreTrainedModel",
    "RnaTokenizer",
    "UtrLmForMaskedLM",
    "UtrLmForSequenceClassification",
    "UtrLmForTokenClassification",
    "UtrLmForNucleotideClassification",
]

AutoConfig.register("utrlm", UtrLmConfig)
AutoModel.register(UtrLmConfig, UtrLmModel)
AutoModelForMaskedLM.register(UtrLmConfig, UtrLmForMaskedLM)
AutoModelForPreTraining.register(UtrLmConfig, UtrLmForPretraining)
AutoModelForSequenceClassification.register(UtrLmConfig, UtrLmForSequenceClassification)
AutoModelForTokenClassification.register(UtrLmConfig, UtrLmForTokenClassification)
AutoModelForNucleotideClassification.register(UtrLmConfig, UtrLmForNucleotideClassification)
AutoTokenizer.register(UtrLmConfig, RnaTokenizer)
