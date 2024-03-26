from transformers import AutoConfig, AutoModel, AutoTokenizer

from multimolecule.tokenizers.rna import RnaTokenizer

from .configuration_rnabert import RnaBertConfig
from .modeling_rnabert import RnaBertModel

__all__ = ["RnaBertConfig", "RnaBertModel", "RnaTokenizer"]

AutoConfig.register("rnabert", RnaBertConfig)
AutoModel.register(RnaBertConfig, RnaBertModel)
AutoTokenizer.register(RnaBertConfig, RnaTokenizer)
