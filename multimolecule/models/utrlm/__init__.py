from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from multimolecule.tokenizers.rna import RnaTokenizer

from .configuration_utrlm import UtrLmConfig
from .modeling_utrlm import (
    UtrLmForMaskedLM,
    UtrLmForPretraining,
    UtrLmForSequenceClassification,
    UtrLmForTokenClassification,
    UtrLmModel,
)

__all__ = [
    "UtrLmConfig",
    "UtrLmModel",
    "RnaTokenizer",
    "UtrLmForMaskedLM",
    "UtrLmForSequenceClassification",
    "UtrLmForTokenClassification",
]

AutoConfig.register("utrlm", UtrLmConfig)
AutoModel.register(UtrLmConfig, UtrLmModel)
AutoModelForMaskedLM.register(UtrLmConfig, UtrLmForMaskedLM)
AutoModelForPreTraining.register(UtrLmConfig, UtrLmForPretraining)
AutoModelForSequenceClassification.register(UtrLmConfig, UtrLmForSequenceClassification)
AutoModelForTokenClassification.register(UtrLmConfig, UtrLmForTokenClassification)
AutoTokenizer.register(UtrLmConfig, RnaTokenizer)
