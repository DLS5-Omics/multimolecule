from collections import OrderedDict

import transformers
import transformers.models
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


class AutoModelForNucleotideClassification(_BaseAutoModelClass):
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())


transformers.models.auto.modeling_auto.AutoModelForNucleotideClassification = AutoModelForNucleotideClassification
transformers.AutoModelForNucleotideClassification = AutoModelForNucleotideClassification
