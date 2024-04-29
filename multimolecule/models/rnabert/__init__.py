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

from .configuration_rnabert import RnaBertConfig
from .modeling_rnabert import (
    RnaBertForMaskedLM,
    RnaBertForNucleotideClassification,
    RnaBertForPretraining,
    RnaBertForSequenceClassification,
    RnaBertForTokenClassification,
    RnaBertModel,
    RnaBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "RnaBertConfig",
    "RnaBertModel",
    "RnaBertPreTrainedModel",
    "RnaBertForMaskedLM",
    "RnaBertForPretraining",
    "RnaBertForSequenceClassification",
    "RnaBertForTokenClassification",
    "RnaBertForNucleotideClassification",
]

AutoConfig.register("rnabert", RnaBertConfig)
AutoModel.register(RnaBertConfig, RnaBertModel)
AutoModelForMaskedLM.register(RnaBertConfig, RnaBertForMaskedLM)
AutoModelForPreTraining.register(RnaBertConfig, RnaBertForPretraining)
AutoModelForSequenceClassification.register(RnaBertConfig, RnaBertForSequenceClassification)
AutoModelForTokenClassification.register(RnaBertConfig, RnaBertForTokenClassification)
# AutoModelForNucleotideClassification.register(RnaBertConfig, RnaBertForNucleotideClassification)
AutoTokenizer.register(RnaBertConfig, RnaTokenizer)
