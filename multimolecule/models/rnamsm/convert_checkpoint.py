# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import os

import __main__
import chanfig
import torch

from multimolecule.models import RnaMsmConfig as Config
from multimolecule.models import RnaMsmForPretraining as Model
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

# eval hack
__main__.OptimizerConfig = chanfig.FlatDict()
__main__.MSATransformerModelConfig = chanfig.FlatDict()
__main__.DataConfig = chanfig.FlatDict()
__main__.TrainConfig = chanfig.FlatDict()
__main__.LoggingConfig = chanfig.FlatDict()


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list):
    state_dict = {}
    for key, value in original_state_dict.items():
        key = key.replace("layers", "rnamsm.encoder.layer")
        key = key.replace("msa_position_embedding", "rnamsm.embeddings.msa_embeddings")
        key = key.replace("embed_tokens", "rnamsm.embeddings.word_embeddings")
        key = key.replace("embed_positions", "rnamsm.embeddings.position_embeddings")
        key = key.replace("emb_layer_norm_before", "rnamsm.embeddings.layer_norm")
        key = key.replace("emb_layer_norm_after", "rnamsm.encoder.layer_norm")
        key = key.replace("regression", "decoder")
        key = key.replace("contact_head", "pretrain_head.contact")
        key = key.replace("lm_head", "pretrain_head.predictions")
        key = key.replace("pretrain_head.predictions.weight", "pretrain_head.predictions.decoder.weight")
        key = key.replace("pretrain_head.predictions.dense", "pretrain_head.predictions.transform.dense")
        key = key.replace("pretrain_head.predictions.layer_norm", "pretrain_head.predictions.transform.layer_norm")
        state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["rnamsm.embeddings.word_embeddings.weight"],
        state_dict["pretrain_head.predictions.decoder.weight"],
        state_dict["pretrain_head.predictions.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rnamsm.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["pretrain_head.predictions.decoder.weight"] = decoder_weight
    state_dict["pretrain_head.predictions.decoder.bias"] = state_dict["pretrain_head.predictions.bias"] = decoder_bias
    return state_dict


def convert_checkpoint(convert_config):
    vocab_list = get_vocab_list()
    original_vocab_list = ["<cls>", "<pad>", "<eos>", "<unk>", "A", "G", "C", "U", "X", "N", "-", "<mask>"]
    config = Config(num_labels=1)
    config.architectures = ["RnaMsmModel"]
    config.vocab_size = len(vocab_list)

    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    ckpt = ckpt.get("state_dict", ckpt)
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
    output_path: str = Config.model_type
    push_to_hub: bool = False
    delete_existing: bool = False
    repo_id: str | None = None
    token: str | None = None

    def post(self):
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
