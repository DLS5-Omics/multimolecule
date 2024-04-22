from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from multimolecule.tokenizers.rna import RnaTokenizer

from .configuration_utrbert import UtrBertConfig
from .modeling_utrbert import (
    UtrBertForMaskedLM,
    UtrBertForPretraining,
    UtrBertForSequenceClassification,
    UtrBertForTokenClassification,
    UtrBertModel,
    UtrBertPreTrainedModel,
)

__all__ = [
    "UtrBertConfig",
    "UtrBertModel",
    "UtrBertPreTrainedModel",
    "UtrBertForMaskedLM",
    "UtrBertForPretraining",
    "UtrBertForSequenceClassification",
    "UtrBertForTokenClassification",
    "UtrBertForCrisprOffTarget",
]

AutoConfig.register("utrbert", UtrBertConfig)
AutoModel.register(UtrBertConfig, UtrBertModel)
AutoModelForMaskedLM.register(UtrBertConfig, UtrBertForMaskedLM)
AutoModelForPreTraining.register(UtrBertConfig, UtrBertForPretraining)
AutoModelForSequenceClassification.register(UtrBertConfig, UtrBertForSequenceClassification)
AutoModelForTokenClassification.register(UtrBertConfig, UtrBertForTokenClassification)
AutoModelWithLMHead.register(UtrBertConfig, UtrBertForTokenClassification)
AutoTokenizer.register(UtrBertConfig, RnaTokenizer)
