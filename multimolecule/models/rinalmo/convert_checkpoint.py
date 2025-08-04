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
from pathlib import Path

import torch

from multimolecule.models import RiNALMoConfig as Config
from multimolecule.models import RiNALMoForPreTraining, RiNALMoForSecondaryStructurePrediction
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet

torch.manual_seed(1016)


def convert_checkpoint(convert_config: ConvertConfig):
    print(f"Converting RiNALMo checkpoint at {convert_config.checkpoint_path}")
    config = get_config(convert_config)
    vocab_list = get_alphabet().vocabulary
    config.vocab_size = len(vocab_list)
    if convert_config.task is None:
        Model = RiNALMoForPreTraining
    elif convert_config.task == "ss":
        Model = RiNALMoForSecondaryStructurePrediction
    else:
        raise ValueError(f"Unknown fine-tuning task: {convert_config.task}")
    model = Model(config)

    ckpt = torch.load(convert_config.checkpoint_path, map_location=torch.device("cpu"))
    ckpt = ckpt.get("model", ckpt)
    state_dict = _convert_checkpoint(config, ckpt, vocab_list, original_vocab_list, task=convert_config.task)

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict, vocab_list, original_vocab_list, task: str | None = None):
    state_dict = {}
    for key, value in original_state_dict.items():
        if "inv_freq" in key or key in {"threshold"}:
            continue
        key = key.replace("lm.", "")
        key = key.replace("gamma", "weight")
        key = key.replace("beta", "bias")
        key = key.replace("embedding", "rinalmo.embeddings.word_embeddings")
        key = key.replace("transformer", "rinalmo")
        key = key.replace("blocks", "encoder.layer")
        key = key.replace("mh_attn", "attention")
        key = key.replace("attn_layer_norm", "attention.layer_norm")
        key = key.replace("out_proj", "output.dense")
        key = key.replace("out_layer_norm", "layer_norm")
        key = key.replace("transition.0", "intermediate")
        key = key.replace("transition.2", "output.dense")
        key = key.replace("final_layer_norm", "encoder.layer_norm")
        key = key.replace("lm_mask_head.linear1", "lm_head.transform.dense")
        key = key.replace("lm_mask_head.layer_norm", "lm_head.transform.layer_norm")
        key = key.replace("lm_mask_head.linear2", "lm_head.decoder")
        if task == "ss":
            key = key.replace("pred_head.linear_in", "ss_head.projection")
            key = key.replace("pred_head.resnet.encoder.layer", "ss_head.convnet")
            key = key.replace("conv_net.0", "conv1")
            key = key.replace("conv_net.3", "conv2")
            key = key.replace("conv_net.6", "conv3")
            key = key.replace("pred_head.conv_out", "ss_head.prediction")
        if "Wqkv" in key:
            q, k, v = (
                key.replace("Wqkv", "self.query"),
                key.replace("Wqkv", "self.key"),
                key.replace("Wqkv", "self.value"),
            )
            state_dict[q], state_dict[k], state_dict[v] = value.chunk(3, dim=0)
        else:
            state_dict[key] = value

    word_embed_weight, decoder_weight, decoder_bias = convert_word_embeddings(
        state_dict["rinalmo.embeddings.word_embeddings.weight"],
        state_dict["lm_head.decoder.weight"],
        state_dict["lm_head.decoder.bias"],
        old_vocab=original_vocab_list,
        new_vocab=vocab_list,
        std=config.initializer_range,
    )
    state_dict["rinalmo.embeddings.word_embeddings.weight"] = word_embed_weight
    state_dict["lm_head.decoder.weight"] = decoder_weight
    state_dict["lm_head.decoder.bias"] = state_dict["lm_head.bias"] = decoder_bias
    return state_dict


original_vocab_list = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "<mask>",
    "A",
    "C",
    "G",
    "U",
    "I",
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
]


def get_config(convert_config: ConvertConfig) -> Config:
    if convert_config.size == "giga":
        config = Config()
    elif convert_config.size == "mega":
        config = Config(
            num_hidden_layers=30,
            hidden_size=640,
            intermediate_size=2560,
        )
    elif convert_config.size == "micro":
        config = Config(
            num_hidden_layers=12,
            hidden_size=480,
            intermediate_size=1920,
        )
    elif convert_config.size == "nano":
        config = Config(
            num_hidden_layers=6,
            hidden_size=320,
            intermediate_size=1280,
        )
    else:
        raise ValueError(f"Unknown size: {convert_config.size}")
    config.architectures = ["RiNALMoModel"]
    return config


class ConvertConfig(ConvertConfig_):
    size: str = "giga"
    task: str | None = None
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type

    def post(self):
        checkpoint_path = Path(self.checkpoint_path).stem.lower()
        if "giga" in checkpoint_path:
            self.size = "giga"
        elif "mega" in checkpoint_path:
            self.size = "mega"
        elif "micro" in checkpoint_path:
            self.size = "micro"
        elif "nano" in checkpoint_path:
            self.size = "nano"
        else:
            raise ValueError(f"Unknown checkpoint size in {self.checkpoint_path}")
        self.output_path += f"-{self.size}"
        if "pretrained" in checkpoint_path:
            self.task = None
        elif "ft" in checkpoint_path:
            if "ss" in checkpoint_path:
                self.task = "ss"
            elif "ncrna" in checkpoint_path:
                self.task = "ncrna"
            elif "mrl" in checkpoint_path:
                self.task = "mrl"
            elif "acceptor" in checkpoint_path:
                self.task = "acceptor"
            elif "donor" in checkpoint_path:
                self.task = "donor"
            else:
                raise ValueError(f"Unknown fine-tuning task in {self.checkpoint_path}")
            self.output_path += f"-{self.task}"
        else:
            raise ValueError(f"Unknown checkpoint type in {self.checkpoint_path}")
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
