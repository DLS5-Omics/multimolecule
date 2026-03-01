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

from multimolecule.models import DnaBertConfig as Config
from multimolecule.models import DnaBertForMaskedLM as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


def _detect_nmers(checkpoint_path: str) -> int:
    """Detect k-mer size from checkpoint path or vocab.txt."""
    basename = os.path.basename(checkpoint_path.rstrip("/"))
    kmer_map = {"DNA_bert_3": 3, "DNA_bert_4": 4, "DNA_bert_5": 5, "DNA_bert_6": 6}
    if basename in kmer_map:
        return kmer_map[basename]
    # Fallback: infer from first non-special token in vocab.txt
    vocab_path = os.path.join(checkpoint_path, "vocab.txt")
    with open(vocab_path) as f:
        for line in f:
            token = line.strip()
            if token and not token.startswith("["):
                return len(token)
    raise ValueError(f"Cannot detect nmers from checkpoint at {checkpoint_path}")


def _read_original_vocab(checkpoint_path: str) -> list[str]:
    """Read original vocab.txt and remap special tokens to multimolecule convention."""
    vocab_path = os.path.join(checkpoint_path, "vocab.txt")
    vocab = []
    with open(vocab_path) as f:
        for line in f:
            token = line.strip()
            if token.startswith("[") and token.endswith("]"):
                token = token.replace("[", "<").replace("]", ">").lower()
            if token == "<sep>":
                token = "<eos>"
            vocab.append(token)
    return vocab


def convert_checkpoint(convert_config):
    nmers = _detect_nmers(convert_config.checkpoint_path)
    print(f"Converting DNABERT {nmers}-mer checkpoint at {convert_config.checkpoint_path}")

    config = chanfig.load(os.path.join(convert_config.checkpoint_path, "config.json"))
    config.hidden_dropout = config.pop("hidden_dropout_prob", 0.1)
    config.attention_dropout = config.pop("attention_probs_dropout_prob", 0.1)
    # Remove fields not part of DnaBertConfig
    for field in (
        "auto_map",
        "classifier_dropout",
        "gradient_checkpointing",
        "torch_dtype",
        "dtype",
        "rnn",
        "rnn_hidden",
        "rnn_dropout",
        "num_rnn_layer",
        "split",
        "do_sample",
        "eos_token_ids",
        "finetuning_task",
        "id2label",
        "label2id",
        "length_penalty",
        "max_length",
        "model_type",
        "num_beams",
        "num_labels",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_past",
        "pruned_heads",
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
        "torchscript",
        "use_bfloat16",
    ):
        config.pop(field, None)

    alphabet = get_alphabet("nucleobase", nmers=nmers)
    new_vocab = list(alphabet.vocabulary)
    config.vocab_size = len(new_vocab)
    config = Config.from_dict(config)
    del config._name_or_path
    config.architectures = ["DnaBertForMaskedLM"]

    model = Model(config)

    ckpt = torch.load(
        os.path.join(convert_config.checkpoint_path, "pytorch_model.bin"), map_location=torch.device("cpu")
    )

    original_vocab = _read_original_vocab(convert_config.checkpoint_path)
    state_dict = _convert_checkpoint(config, ckpt, original_vocab, new_vocab)
    load_checkpoint(model, state_dict)

    tokenizer_config = get_tokenizer_config()
    tokenizer_config["nmers"] = nmers
    tokenizer_config["alphabet"] = "nucleobase"
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, original_vocab, new_vocab):
    state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key.replace("LayerNorm", "layer_norm")
        new_key = new_key.replace("gamma", "weight")
        new_key = new_key.replace("beta", "bias")

        if new_key.startswith("bert"):
            new_key = "model" + new_key[4:]
            state_dict[new_key] = value
            continue

        if new_key.startswith("cls"):
            new_key = "lm_head" + new_key[15:]
            state_dict[new_key] = value
            if new_key == "lm_head.decoder.bias":
                state_dict["lm_head.bias"] = value
            continue

        state_dict[new_key] = value

    # Remap word embeddings for vocabulary change
    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["model.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab,
        new_vocab=new_vocab,
        std=config.initializer_range,
    )
    state_dict["model.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = decoder_bias
    state_dict["lm_head.bias"] = decoder_bias

    # Remove keys not present in ForMaskedLM
    state_dict.pop("model.embeddings.position_ids", None)
    state_dict.pop("model.pooler.dense.weight", None)
    state_dict.pop("model.pooler.dense.bias", None)

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]
    default_variant: str | None = "dnabert"

    def post(self):
        if self.output_path is None:
            nmers = _detect_nmers(self.checkpoint_path)
            self.output_path = f"dnabert-{nmers}mer"
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
