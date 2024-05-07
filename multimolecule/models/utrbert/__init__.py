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
from .configuration_utrbert import UtrBertConfig
from .modeling_utrbert import (
    UtrBertForMaskedLM,
    UtrBertForNucleotideClassification,
    UtrBertForPretraining,
    UtrBertForSequenceClassification,
    UtrBertForTokenClassification,
    UtrBertModel,
    UtrBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertPreTrainedModel",
    "UtrBertForMaskedLM",
    "UtrBertForPretraining",
    "UtrBertForSequenceClassification",
    "UtrBertForTokenClassification",
    "UtrBertForNucleotideClassification",
]

AutoConfig.register("utrbert", UtrBertConfig)
AutoModel.register(UtrBertConfig, UtrBertModel)
AutoModelForMaskedLM.register(UtrBertConfig, UtrBertForMaskedLM)
AutoModelForPreTraining.register(UtrBertConfig, UtrBertForPretraining)
AutoModelForSequenceClassification.register(UtrBertConfig, UtrBertForSequenceClassification)
AutoModelForTokenClassification.register(UtrBertConfig, UtrBertForTokenClassification)
AutoModelForNucleotideClassification.register(UtrBertConfig, UtrBertForNucleotideClassification)
AutoTokenizer.register(UtrBertConfig, RnaTokenizer)
