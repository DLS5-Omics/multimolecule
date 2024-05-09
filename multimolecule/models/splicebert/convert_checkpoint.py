from __future__ import annotations

import os

import chanfig
import torch

from multimolecule.models import SpliceBertConfig as Config
from multimolecule.models import SpliceBertForPretraining as Model
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
        key = key.replace("LayerNorm", "layer_norm")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        if key.startswith("bert"):
            state_dict["splice" + key] = value
            continue
        if key.startswith("cls"):
            key = "lm_head" + key[15:]
            state_dict[key] = value
            continue
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["splicebert.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["splicebert.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    del state_dict["splicebert.embeddings.position_ids"]
    return state_dict


def convert_checkpoint(convert_config):
    config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config.hidden_dropout = config.pop("hidden_dropout_prob", 0.1)
    config.attention_dropout = config.pop("attention_probs_dropout_prob", 0.1)
    vocab_list = get_vocab_list()
    config = Config.from_dict(config)
    del config._name_or_path
    config.architectures = ["SpliceBertModel"]
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(
        os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
    )
    original_vocab_list = []
    for char in open(os.path.join(convert_config.checkpoint_path, "vocab.txt")).read().splitlines():  # noqa: SIM115
        if char.startswith("["):
            char = char.lower().replace("[", "<").replace("]", ">")
        if char == "T":
            char = "U"
        if char == "<sep>":
            char = "<eos>"
        original_vocab_list.append(char)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list)

    model.load_state_dict(state_dict)
    model.save_pretrained(convert_config.output_path, safe_serialization=True)
    model.save_pretrained(convert_config.output_path, safe_serialization=False)
    chanfig.NestedDict(get_special_tokens_map()).json(
        os.path.join(convert_config.output_path, "special_tokens_map.json")
    )
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["model_max_length"] = config.max_position_embeddings
    tokenizer_config.json(os.path.join(convert_config.output_path, "tokenizer_config.json"))

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
    output_path: str | None = None
    push_to_hub: bool = False
    delete_existing: bool = False
    repo_id: str | None = None
    token: str | None = None

    def post(self):
        if self.output_path is None:
            self.output_path = self.checkpoint_path.split("/")[-1].lower()
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
