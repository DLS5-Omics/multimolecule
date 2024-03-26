from transformers import AutoTokenizer

from . import models, tokenizers
from .models import RnaBertConfig
from .tokenizers import RnaTokenizer

AutoTokenizer.register(RnaBertConfig, RnaTokenizer)


__all__ = ["models", "tokenizers"]
