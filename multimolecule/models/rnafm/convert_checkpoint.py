import os
from typing import Optional

import chanfig
import torch
from torch import nn

from multimolecule.models import RnaFmConfig as Config
from multimolecule.models import RnaFmForPretraining as Model
from multimolecule.tokenizers.rna.utils import get_special_tokens_map, get_tokenizer_config, get_vocab_list

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

torch.manual_seed(1013)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = "rnafm" + key[7:]
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("rnafm.encoder.emb_layer_norm_before", "rnafm.embeddings.layer_norm")
        key = key.replace("rnafm.encoder.embed_tokens", "rnafm.embeddings.word_embeddings")
        key = key.replace("rnafm.encoder.embed_positions", "rnafm.embeddings.position_embeddings")
        key = key.replace("layers", "layer")
        key = key.replace("self_attn", "attention.self")
        key = key.replace("q_proj", "query")
        key = key.replace("k_proj", "key")
        key = key.replace("v_proj", "value")
        key = key.replace("self.out_proj", "output.dense")
        key = key.replace("self_layer_norm", "layer_norm")
        key = key.replace("final_layer_norm", "layer_norm")
        key = key.replace("fc1", "intermediate.dense")
        key = key.replace("fc2", "output.dense")
        key = key.replace("regression", "decoder")
        key = key.replace("rnafm.encoder.lm_head", "pretrain_head.predictions")
        key = key.replace("predictions.dense", "predictions.transform.dense")
        key = key.replace("predictions.layer_norm", "predictions.transform.layer_norm")
        key = key.replace("predictions.weight", "predictions.decoder.weight")
        key = key.replace("rnafm.encoder.contact_head", "pretrain_head.contact")
        state_dict[key] = value

    state_vocab_size = state_dict["rnafm.embeddings.word_embeddings.weight"].size(0)
    original_vocab_size = len(original_vocab_list)
    if state_vocab_size != original_vocab_size:
        raise ValueError(
            f"Vocabulary size do not match. Expected to have {original_vocab_size}, but got {state_vocab_size}."
        )
    word_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    word_embed_weight = word_embed.weight.data
    predictions_decoder_weight = torch.zeros((config.vocab_size, config.hidden_size))
    predictions_bias = torch.zeros(config.vocab_size)
    # nn.init.normal_(pos_embed.weight, std=0.02)
    for original_index, original_token in enumerate(original_vocab_list):
        new_index = vocab_list.index(original_token)
        word_embed_weight[new_index] = state_dict["rnafm.embeddings.word_embeddings.weight"][original_index]
        predictions_decoder_weight[new_index] = state_dict["pretrain_head.predictions.decoder.weight"][original_index]
        predictions_bias[new_index] = state_dict["pretrain_head.predictions.bias"][original_index]
    state_dict["rnafm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["pretrain_head.predictions.decoder.weight"] = predictions_decoder_weight
    state_dict["pretrain_head.predictions.decoder.bias"] = state_dict["pretrain_head.predictions.bias"] = (
        predictions_bias
    )
    return state_dict


def convert_checkpoint(convert_config):
    vocab_list = get_vocab_list()
    original_vocab_list = [
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "A",
        "C",
        "G",
        "U",
        "R",
        "Y",
        "K",
        "M",
        "S",
        "W",
        "B",
        "D",
        "H",
        "V",
        "N",
        "-",
        "<null>",
        "<null>",
        "<null>",
        "<null>",
        "<mask>",
    ]
    config = Config(num_labels=1)
    config.architectures = ["RnaFmModel"]
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)
    model.save_pretrained(convert_config.output_path, safe_serialization=True)
    model.save_pretrained(convert_config.output_path, safe_serialization=False)
    chanfig.NestedDict(get_special_tokens_map()).json(
        os.path.join(convert_config.output_path, "special_tokens_map.json")
    )
    chanfig.NestedDict(get_tokenizer_config()).json(os.path.join(convert_config.output_path, "tokenizer_config.json"))

    if convert_config.push_to_hub:
        if HfApi is None:
            raise ImportError("Please install huggingface_hub to push to the hub.")
        api = HfApi()
        api.create_repo(
            convert_config.repo_id,
            token=convert_config.token,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=convert_config.repo_id, folder_path=convert_config.output_path, token=convert_config.token
        )


@chanfig.configclass
class ConvertConfig:
    checkpoint_path: str
    output_path: str = Config.model_type
    push_to_hub: bool = False
    repo_id: str = f"multimolecule/{output_path}"
    token: Optional[str] = None


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
