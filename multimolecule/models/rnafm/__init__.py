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
from .configuration_rnafm import RnaFmConfig
from .modeling_rnafm import (
    RnaFmForMaskedLM,
    RnaFmForNucleotideClassification,
    RnaFmForPretraining,
    RnaFmForSequenceClassification,
    RnaFmForTokenClassification,
    RnaFmModel,
    RnaFmPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaFmConfig",
    "RnaFmModel",
    "RnaFmPreTrainedModel",
    "RnaFmForMaskedLM",
    "RnaFmForPretraining",
    "RnaFmForSequenceClassification",
    "RnaFmForTokenClassification",
    "RnaFmForNucleotideClassification",
]

AutoConfig.register("rnafm", RnaFmConfig)
AutoModel.register(RnaFmConfig, RnaFmModel)
AutoModelForMaskedLM.register(RnaFmConfig, RnaFmForMaskedLM)
AutoModelForPreTraining.register(RnaFmConfig, RnaFmForPretraining)
AutoModelForSequenceClassification.register(RnaFmConfig, RnaFmForSequenceClassification)
AutoModelForTokenClassification.register(RnaFmConfig, RnaFmForTokenClassification)
AutoModelForNucleotideClassification.register(RnaFmConfig, RnaFmForNucleotideClassification)
AutoTokenizer.register(RnaFmConfig, RnaTokenizer)
