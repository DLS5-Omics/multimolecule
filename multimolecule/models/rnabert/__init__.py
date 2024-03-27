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

from .configuration_rnabert import RnaBertConfig
from .modeling_rnabert import (
    RnaBertForMaskedLM,
    RnaBertForPretraining,
    RnaBertForSequenceClassification,
    RnaBertForTokenClassification,
    RnaBertModel,
)

__all__ = [
    "RnaBertConfig",
    "RnaBertModel",
    "RnaTokenizer",
    "RnaBertForMaskedLM",
    "RnaBertForPretraining",
    "RnaBertForSequenceClassification",
    "RnaBertForTokenClassification",
]

AutoConfig.register("rnabert", RnaBertConfig)
AutoModel.register(RnaBertConfig, RnaBertModel)
AutoModelForMaskedLM.register(RnaBertConfig, RnaBertForMaskedLM)
AutoModelForPreTraining.register(RnaBertConfig, RnaBertForPretraining)
AutoModelForSequenceClassification.register(RnaBertConfig, RnaBertForSequenceClassification)
AutoModelForTokenClassification.register(RnaBertConfig, RnaBertForTokenClassification)
AutoTokenizer.register(RnaBertConfig, RnaTokenizer)
