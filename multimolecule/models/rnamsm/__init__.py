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

from .configuration_rnamsm import RnaMsmConfig
from .modeling_rnamsm import (
    RnaMsmForMaskedLM,
    RnaMsmForPretraining,
    RnaMsmForSequenceClassification,
    RnaMsmForTokenClassification,
    RnaMsmModel,
    RnaMsmPreTrainedModel,
)

__all__ = [
    "RnaMsmConfig",
    "RnaMsmModel",
    "RnaTokenizer",
    "RnaMsmPreTrainedModel",
    "RnaMsmForPretraining",
    "RnaMsmForMaskedLM",
    "RnaMsmForSequenceClassification",
    "RnaMsmForTokenClassification",
]

AutoConfig.register("rnamsm", RnaMsmConfig)
AutoModel.register(RnaMsmConfig, RnaMsmModel)
AutoModelForMaskedLM.register(RnaMsmConfig, RnaMsmForMaskedLM)
AutoModelForPreTraining.register(RnaMsmConfig, RnaMsmForPretraining)
AutoModelForSequenceClassification.register(RnaMsmConfig, RnaMsmForSequenceClassification)
AutoModelForTokenClassification.register(RnaMsmConfig, RnaMsmForTokenClassification)
AutoTokenizer.register(RnaMsmConfig, RnaTokenizer)
