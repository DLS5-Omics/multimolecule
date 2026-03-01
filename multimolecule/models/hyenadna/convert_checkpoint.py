# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


from __future__ import annotations

import os

import chanfig
import torch
from safetensors.torch import load_file

from multimolecule.models import HyenaDnaConfig as Config
from multimolecule.models import HyenaDnaForCausalLM as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Original HyenaDNA vocab:
# [CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5, [UNK]=6, A=7, C=8, G=9, T=10, N=11
ORIGINAL_VOCAB = ["[CLS]", "[SEP]", "[BOS]", "[MASK]", "[PAD]", "[RESERVED]", "[UNK]", "A", "C", "G", "T", "N"]

# Map original special tokens to multimolecule convention
VOCAB_REMAP = {
    "[CLS]": "<cls>",
    "[SEP]": "<eos>",
    "[BOS]": "<cls>",
    "[MASK]": "<mask>",
    "[PAD]": "<pad>",
    "[RESERVED]": "<null>",
    "[UNK]": "<unk>",
}


def _get_original_vocab() -> list[str]:
    """Get the original HyenaDNA vocabulary with special tokens remapped."""
    return [VOCAB_REMAP.get(tok, tok) for tok in ORIGINAL_VOCAB]


def convert_checkpoint(convert_config):
    print(f"Converting HyenaDNA checkpoint at {convert_config.checkpoint_path}")

    config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))

    # Map original config field names to our convention
    mm_config = {
        "vocab_size": config.get("vocab_size", 12),
        "hidden_size": config.get("d_model", 256),
        "num_hidden_layers": config.get("n_layer", 8),
        "intermediate_size": config.get("d_inner"),
        "embedding_dropout": config.get("embed_dropout", 0.1),
        "hidden_dropout": config.get("hyena_dropout", 0.0),
        "max_position_embeddings": config.get("max_seq_len", 1024),
        "initializer_range": config.get("initializer_range", 0.02),
        "layer_norm_eps": config.get("layer_norm_epsilon", 1e-5),
        "hyena_order": config.get("hyena_order", 2),
        "filter_order": config.get("filter_order", 64),
        "short_filter_order": config.get("short_filter_order", 3),
        "filter_emb_dim": config.get("emb_dim", 5),
        "num_inner_mlps": config.get("num_inner_mlps", 2),
        "activation_freq": config.get("activation_freq", 10),
        "filter_dropout": config.get("hyena_filter_dropout", 0.0),
        "use_bias": config.get("use_bias", True),
        "train_freq": config.get("train_freq", True),
        "pad_vocab_size_multiple": config.get("pad_vocab_size_multiple", 8),
    }

    # Build new vocab from DnaTokenizer's streamline alphabet (includes N)
    alphabet = get_alphabet("streamline")
    new_vocab = list(alphabet.vocabulary)
    mm_config["vocab_size"] = len(new_vocab)

    mm_config = Config.from_dict(mm_config)
    del mm_config._name_or_path
    mm_config.architectures = ["HyenaDnaForCausalLM"]

    model = Model(mm_config)

    # Load checkpoint
    safetensors_path = os.path.join(convert_config.checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        ckpt = load_file(safetensors_path)
    else:
        ckpt = torch.load(
            os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
        )

    original_vocab = _get_original_vocab()
    state_dict = _convert_checkpoint(mm_config, ckpt, original_vocab, new_vocab)
    load_checkpoint(model, state_dict)

    tokenizer_config = get_tokenizer_config()
    tokenizer_config["alphabet"] = "streamline"
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, original_vocab, new_vocab):
    state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key
        # hyena.backbone.* -> model.*
        if new_key.startswith("hyena.backbone."):
            new_key = "model." + new_key[len("hyena.backbone.") :]

        # ln_f -> final_layer_norm
        new_key = new_key.replace(".ln_f.", ".final_layer_norm.")
        if new_key.endswith(".ln_f.weight"):
            new_key = new_key.replace(".ln_f.weight", ".final_layer_norm.weight")
        elif new_key.endswith(".ln_f.bias"):
            new_key = new_key.replace(".ln_f.bias", ".final_layer_norm.bias")

        state_dict[new_key] = value

    # The original HyenaDNA uses a single shared HyenaSin activation instance across
    # all positions in the implicit_filter Sequential. The original checkpoint only stores
    # the freq parameter once (at index 1). Our model registers it at all positions (1, 3, 5, ...),
    # so we need to duplicate the freq values for the shared positions.
    import re

    freq_pattern = re.compile(r"(model\.layers\.\d+\.mixer\.filter_fn\.implicit_filter\.)1(\.freq)")
    freq_keys = [(k, v) for k, v in state_dict.items() if freq_pattern.match(k)]
    for key, value in freq_keys:
        match = freq_pattern.match(key)
        prefix, suffix = match.group(1), match.group(2)
        for idx in range(3, 3 + 2 * config.num_inner_mlps, 2):
            state_dict[f"{prefix}{idx}{suffix}"] = value.clone()

    # Remap word embeddings for vocabulary change
    embed_key = "model.embeddings.word_embeddings.weight"
    lm_head_key = "lm_head.weight"

    if embed_key in state_dict:
        # Pad original vocab to match checkpoint's padded size
        padded_original_vocab = list(original_vocab)
        checkpoint_vocab_size = state_dict[embed_key].shape[0]
        while len(padded_original_vocab) < checkpoint_vocab_size:
            padded_original_vocab.append(f"<unused{len(padded_original_vocab)}>")

        # Pad new vocab to match model's padded size
        padded_new_vocab = list(new_vocab)
        new_padded_size = config.vocab_size
        if new_padded_size % config.pad_vocab_size_multiple != 0:
            new_padded_size += config.pad_vocab_size_multiple - (new_padded_size % config.pad_vocab_size_multiple)
        while len(padded_new_vocab) < new_padded_size:
            padded_new_vocab.append(f"<unused{len(padded_new_vocab)}>")

        if lm_head_key in state_dict:
            (embed_weight, lm_head_weight) = convert_word_embeddings(
                state_dict[embed_key],
                state_dict[lm_head_key],
                old_vocab=padded_original_vocab,
                new_vocab=padded_new_vocab,
                std=config.initializer_range,
            )
            state_dict[embed_key] = embed_weight
            state_dict[lm_head_key] = lm_head_weight
        else:
            (embed_weight,) = convert_word_embeddings(
                state_dict[embed_key],
                old_vocab=padded_original_vocab,
                new_vocab=padded_new_vocab,
                std=config.initializer_range,
            )
            state_dict[embed_key] = embed_weight

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]
    default_variant: str | None = "hyenadna-medium"

    def post(self):
        if self.output_path is None:
            basename = os.path.basename(self.checkpoint_path.rstrip("/")).lower()
            # hyenadna-medium-450k-seqlen-hf -> hyenadna-medium
            self.output_path = "-".join(basename.split("-")[:2])
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
