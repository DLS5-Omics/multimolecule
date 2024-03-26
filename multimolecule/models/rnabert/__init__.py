from transformers import AutoConfig, AutoModel, AutoTokenizer

from .configuration_rnabert import RnaBertConfig
from .modeling_rnabert import RnaBertModel
from .tokenization_rnabert import RnaBertTokenizer

__all__ = ["RnaBertConfig", "RnaBertModel", "RnaBertTokenizer"]

AutoConfig.register("rnabert", RnaBertConfig)
AutoModel.register(RnaBertConfig, RnaBertModel)
AutoTokenizer.register(RnaBertConfig, RnaBertTokenizer)
