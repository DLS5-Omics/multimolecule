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

from .configuration_rnafm import RnaFmConfig
from .modeling_rnafm import (
    RnaFmForMaskedLM,
    RnaFmForPretraining,
    RnaFmForSequenceClassification,
    RnaFmForTokenClassification,
    RnaFmModel,
)

__all__ = [
    "RnaFmConfig",
    "RnaFmModel",
    "RnaTokenizer",
    "RnaFmForMaskedLM",
    "RnaFmForPretraining",
    "RnaFmForSequenceClassification",
    "RnaFmForTokenClassification",
]

AutoConfig.register("rnafm", RnaFmConfig)
AutoModel.register(RnaFmConfig, RnaFmModel)
AutoModelForMaskedLM.register(RnaFmConfig, RnaFmForMaskedLM)
AutoModelForPreTraining.register(RnaFmConfig, RnaFmForPretraining)
AutoModelForSequenceClassification.register(RnaFmConfig, RnaFmForSequenceClassification)
AutoModelForTokenClassification.register(RnaFmConfig, RnaFmForTokenClassification)
AutoTokenizer.register(RnaFmConfig, RnaTokenizer)
