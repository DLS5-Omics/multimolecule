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
from .configuration_splicebert import SpliceBertConfig
from .modeling_splicebert import (
    SpliceBertForMaskedLM,
    SpliceBertForNucleotideClassification,
    SpliceBertForPretraining,
    SpliceBertForSequenceClassification,
    SpliceBertForTokenClassification,
    SpliceBertModel,
    SpliceBertPreTrainedModel,
)

__all__ = [
    "RnaTokenizer",
    "SpliceBertConfig",
    "SpliceBertModel",
    "SpliceBertPreTrainedModel",
    "SpliceBertForMaskedLM",
    "SpliceBertForPretraining",
    "SpliceBertForSequenceClassification",
    "SpliceBertForTokenClassification",
    "SpliceBertForNucleotideClassification",
]

AutoConfig.register("splicebert", SpliceBertConfig)
AutoModel.register(SpliceBertConfig, SpliceBertModel)
AutoModelForMaskedLM.register(SpliceBertConfig, SpliceBertForMaskedLM)
AutoModelForPreTraining.register(SpliceBertConfig, SpliceBertForPretraining)
AutoModelForSequenceClassification.register(SpliceBertConfig, SpliceBertForSequenceClassification)
AutoModelForTokenClassification.register(SpliceBertConfig, SpliceBertForTokenClassification)
AutoModelForNucleotideClassification.register(SpliceBertConfig, SpliceBertForNucleotideClassification)
AutoTokenizer.register(SpliceBertConfig, RnaTokenizer)
