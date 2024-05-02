import os
from typing import Optional

import chanfig
import torch

from multimolecule.models import UtrLmConfig as Config
from multimolecule.models import UtrLmForPretraining as Model
from multimolecule.tokenisers.rna.utils import (
    convert_word_embeddings,
    get_special_tokens_map,
    get_tokenizer_config,
    get_vocab_list,
)

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

torch.manual_seed(1013)


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        key = "utrlm." + key
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("utrlm.encoder.emb_layer_norm_before", "utrlm.embeddings.layer_norm")
        key = key.replace("utrlm.emb_layer_norm_after", "utrlm.encoder.emb_layer_norm_after")
        key = key.replace("utrlm.embed_tokens", "utrlm.embeddings.word_embeddings")
        key = key.replace("rot_emb", "rotary_embeddings")
        key = key.replace("layers", "encoder.layer")
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
        key = key.replace("utrlm.lm_head", "pretrain_head.predictions")
        key = key.replace("predictions.dense", "predictions.transform.dense")
        key = key.replace("predictions.layer_norm", "predictions.transform.layer_norm")
        key = key.replace("predictions.weight", "predictions.decoder.weight")
        key = key.replace("utrlm.contact_head", "pretrain_head.contact")
        key = key.replace("utrlm.structure_linear", "pretrain_head.structure.decoder")
        key = key.replace("utrlm.supervised_linear", "pretrain_head.supervised.decoder")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["utrlm.embeddings.word_embeddings.weight"],
        state_dict["pretrain_head.predictions.decoder.weight"],
        state_dict["pretrain_head.predictions.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["utrlm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["pretrain_head.predictions.decoder.weight"] = decoder_weight
    state_dict["pretrain_head.predictions.decoder.bias"] = state_dict["pretrain_head.predictions.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    config = chanfig.FlatDict()
    config.supervised_head = {"num_labels": 1}
    if "4.1" in convert_config.checkpoint_path:
        config.structure_head = {"num_labels": 3}
    vocab_list = get_vocab_list()
    original_vocab_list = ["<pad>", "<eos>", "<unk>", "A", "G", "C", "U", "<cls>", "<mask>", "<eos>"]
    config = Config.from_dict(config)
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
        if convert_config.delete_existing:
            api.delete_repo(
                convert_config.repo_id,
                token=convert_config.token,
                missing_ok=True,
            )
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
    delete_existing: bool = False
    repo_id: Optional[str] = None
    token: Optional[str] = None

    def post(self):
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)