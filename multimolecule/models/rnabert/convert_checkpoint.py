import os
import sys
from typing import Optional

import chanfig
import torch
from torch import nn

from multimolecule.models import RnaBertConfig, RnaBertModel
from multimolecule.tokenizers.rna.utils import get_special_tokens_map, get_tokenizer_config, get_vocab_list

CONFIG = {
    "architectures": ["RnaBertModel"],
    "attention_probs_dropout_prob": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 120,
    "intermediate_size": 40,
    "max_position_embeddings": 440,
    "num_attention_heads": 12,
    "num_hidden_layers": 6,
    "vocab_size": 25,
    "ss_vocab_size": 8,
    "type_vocab_size": 2,
    "pad_token_id": 0,
}

original_vocab_list = ["<pad>", "<mask>", "A", "U", "G", "C"]
vocab_list = get_vocab_list()


def convert_checkpoint(checkpoint_path: str, output_path: Optional[str] = None):
    if output_path is None:
        output_path = "rnabert"
    config = RnaBertConfig.from_dict(chanfig.FlatDict(CONFIG))
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
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

    word_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    # nn.init.normal_(pos_embed.weight, std=0.02)
    for original_token, new_token in zip(original_vocab_list, vocab_list):
        original_index = original_vocab_list.index(original_token)
        new_index = vocab_list.index(new_token)
        word_embed.weight.data[new_index] = state_dict["embeddings.word_embeddings.weight"][original_index]
    state_dict["embeddings.word_embeddings.weight"] = word_embed.weight.data

    model.load_state_dict(state_dict)
    model.save_pretrained(output_path, safe_serialization=True)
    model.save_pretrained(output_path, safe_serialization=False)
    chanfig.NestedDict(get_special_tokens_map()).json(os.path.join(output_path, "special_tokens_map.json"))
    chanfig.NestedDict(get_tokenizer_config()).json(os.path.join(output_path, "tokenizer_config.json"))


if __name__ == "__main__":
    convert_checkpoint(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
