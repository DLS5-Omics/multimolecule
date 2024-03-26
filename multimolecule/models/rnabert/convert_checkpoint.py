import sys
from typing import Optional

import chanfig
import torch

from multimolecule.models import RnaBertConfig, RnaBertModel
from multimolecule.models.rnabert.configuration_rnabert import DEFAULT_VOCAB_LIST

CONFIG = {
    "architectures": ["RnaBertModel"],
    "attention_probs_dropout_prob": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 120,
    "initializer_range": 0.02,
    "intermediate_size": 40,
    "layer_norm_eps": 1e-12,
    "mask_token_id": 1,
    "max_position_embeddings": 440,
    "model_type": "rnabert",
    "num_attention_heads": 12,
    "num_hidden_layers": 6,
    "position_embedding_type": "absolute",
    "ss_size": 8,
    "torch_dtype": "float32",
    "type_vocab_size": 2,
}


def convert_checkpoint(checkpoint_path: str, output_path: Optional[str] = None):
    if output_path is None:
        output_path = "rnabert"
    config = RnaBertConfig.from_dict(chanfig.NestedDict(CONFIG))
    config.vocab_list = DEFAULT_VOCAB_LIST
    config.vocab_size = len(config.vocab_list)
    ckpt = torch.load(checkpoint_path)
    bert_state_dict = ckpt
    state_dict = {}

    model = RnaBertModel(config)

    for key, value in bert_state_dict.items():
        if key.startswith("module.cls"):
            continue
        key = key[12:]
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        state_dict[key] = value

    model.load_state_dict(state_dict)
    model.save_pretrained(output_path)


if __name__ == "__main__":
    convert_checkpoint(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
