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

from .configuration_rnafm import RnaFmConfig
from .modeling_rnafm import (
    RnaFmForMaskedLM,
    RnaFmForPretraining,
    RnaFmForSequenceClassification,
    RnaFmForTokenClassification,
    RnaFmModel,
    RnaFmPreTrainedModel,
)

__all__ = [
    "RnaFmConfig",
    "RnaFmModel",
    "RnaTokenizer",
    "RnaFmPreTrainedModel",
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
