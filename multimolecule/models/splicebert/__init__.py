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

from .configuration_splicebert import SpliceBertConfig
from .modeling_splicebert import (
    SpliceBertForMaskedLM,
    SpliceBertForPretraining,
    SpliceBertForSequenceClassification,
    SpliceBertForTokenClassification,
    SpliceBertModel,
)

__all__ = [
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertForMaskedLM",
    "SpliceBertForPretraining",
    "SpliceBertForSequenceClassification",
    "SpliceBertForTokenClassification",
]

AutoConfig.register("splicebert", SpliceBertConfig)
AutoModel.register(SpliceBertConfig, SpliceBertModel)
AutoModelForMaskedLM.register(SpliceBertConfig, SpliceBertForMaskedLM)
AutoModelForPreTraining.register(SpliceBertConfig, SpliceBertForPretraining)
AutoModelForSequenceClassification.register(SpliceBertConfig, SpliceBertForSequenceClassification)
AutoModelForTokenClassification.register(SpliceBertConfig, SpliceBertForTokenClassification)
AutoTokenizer.register(SpliceBertConfig, RnaTokenizer)
