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

# Source notes:
#   Paper:      Chao, Mao, Liu, Salzberg, Pertea. "OpenSpliceAI: An efficient, modular implementation of SpliceAI
#               enabling easy retraining on non-human species." eLife 14:RP107454 (2025).
#   DOI:        10.7554/eLife.107454.3
#   Repository: https://github.com/Kuanhao-Chao/OpenSpliceAI
#   Weights:    OpenSpliceAI bundles ensembles of trained models (human MANE plus retrained species).
#               The canonical human model is `openspliceai-mane` at the 10000nt (full SpliceAI-10k)
#               flanking size. Each ensemble member is a different training random seed (rs10..rs14);
#               ensemble membership is an implementation detail (per implementation_guide.md), so the
#               default checkpoint uses the canonical single member `model_10000nt_rs10.pt`.
#               Other species / flanking sizes / seeds are addable via `--checkpoint_path`.


from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import chanfig
import torch

from multimolecule.models import OpenSpliceAiConfig as Config
from multimolecule.models import OpenSpliceAiForTokenPrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.models.openspliceai.configuration_openspliceai import OpenSpliceAiStageConfig
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)

# Upstream OpenSpliceAI one-hot input channel order maps T to the RNA U channel in MultiMolecule.
ORIGINAL_NUCLEOTIDE_ORDER = ["A", "C", "G", "U"]

StageKind = Literal["block", "skip"]
StageLayout = tuple[int, int, StageKind, int | None]
Variant = tuple[str, int]

DEFAULT_VARIANT = "openspliceai-mane.10000"
DEFAULT_SEED = 10
SPECIES = ("mane", "mouse", "zebrafish", "honeybee", "arabidopsis")
CONTEXTS = (80, 400, 2000, 10000)
CONTEXT_STAGE_SPECS = {
    80: ((4, 11, 1),),
    400: ((4, 11, 1), (4, 11, 4)),
    2000: ((4, 11, 1), (4, 11, 4), (4, 21, 10)),
    10000: ((4, 11, 1), (4, 11, 4), (4, 21, 10), (4, 41, 25)),
}


def _variant_name(species: str, context: int) -> str:
    return f"openspliceai-{species}.{context}"


def _checkpoint_path(checkpoint_root: str, species: str, context: int, seed: int = DEFAULT_SEED) -> str:
    return str(Path(checkpoint_root) / f"openspliceai-{species}" / f"{context}nt" / f"model_{context}nt_rs{seed}.pt")


def _iter_variants() -> list[Variant]:
    return [(species, context) for species in SPECIES for context in CONTEXTS]


def _infer_context(checkpoint_path: str | None, context: int | None) -> int:
    if context is not None:
        if context not in CONTEXT_STAGE_SPECS:
            raise ValueError(f"Unsupported OpenSpliceAI context: {context}. Expected one of {CONTEXTS}.")
        return context
    if checkpoint_path is not None:
        match = re.search(r"(\d+)nt", checkpoint_path)
        if match:
            return _infer_context(None, int(match.group(1)))
    raise ValueError("OpenSpliceAI context is required; pass context=<80|400|2000|10000> or a *_<context>nt_*.pt path.")


def _stages_for_context(context: int) -> list[OpenSpliceAiStageConfig]:
    return [
        OpenSpliceAiStageConfig(num_blocks=num_blocks, kernel_size=kernel_size, dilation=dilation)
        for num_blocks, kernel_size, dilation in CONTEXT_STAGE_SPECS[context]
    ]


def _make_config(context: int) -> Config:
    config = Config(context=context, stages=_stages_for_context(context))
    config.architectures = ["OpenSpliceAiForTokenPrediction"]
    return config


def _stage_layout(config: Config) -> list[StageLayout]:
    """Return, per upstream ``residual_units`` index, a (stage, kind, position) descriptor.

    Upstream interleaves ``ResidualUnit`` and ``Skip`` modules: every 4 residual units is followed
    by a ``Skip``. The trailing ``Skip`` aggregates each stage's contribution into ``context``.
    """
    layout: list[StageLayout] = []
    for stage_idx, stage in enumerate(config.stages):
        for block_idx in range(stage["num_blocks"]):
            layout.append((len(layout), stage_idx, "block", block_idx))
        layout.append((len(layout), stage_idx, "skip", None))
    return layout


def convert_original_state_dict_key(key: str, config: Config) -> str:
    if key.startswith("initial_conv."):
        return key.replace("initial_conv.", "model.encoder.projection.")
    if key.startswith("initial_skip.conv."):
        return key.replace("initial_skip.conv.", "model.encoder.skip.")
    if key.startswith("final_conv."):
        return key.replace("final_conv.", "token_head.decoder.")
    if key.startswith("residual_units."):
        _, idx_str, rest = key.split(".", 2)
        idx = int(idx_str)
        for layout_idx, stage_idx, kind, position in _stage_layout(config):
            if layout_idx != idx:
                continue
            if kind == "skip":
                # `Skip` is a single `conv`; upstream key is `residual_units.<idx>.conv.<param>`.
                return f"model.encoder.stages.{stage_idx}.conv.{rest.split('.', 1)[1]}"
            sub = rest.replace("batchnorm1", "norm1").replace("batchnorm2", "norm2")
            return f"model.encoder.stages.{stage_idx}.blocks.{position}.{sub}"
        raise ValueError(f"Unmapped residual_units index in key: {key}")
    raise ValueError(f"Unexpected upstream key: {key}")


def convert_original_state_dict_value(key: str, value: torch.Tensor, new_vocab: list) -> torch.Tensor:
    # `final_conv` is a 1x1 Conv1d (out, in, 1); the MultiMolecule token head is a Linear (out, in).
    if key.startswith("final_conv.") and key.endswith(".weight"):
        return value.squeeze(-1)
    # First learned layer sees vocabulary-dependent channels; delegate token conversion to the RNA tokenizer rules.
    if key == "initial_conv.weight":
        return convert_one_hot_embeddings(
            value,
            old_vocab=ORIGINAL_NUCLEOTIDE_ORDER,
            new_vocab=new_vocab,
            convert_word_embeddings=convert_word_embeddings,
        )
    return value


def _convert_checkpoint(original_state_dict: OrderedDict, config: Config, new_vocab: list) -> OrderedDict:
    if "initial_conv.weight" not in original_state_dict:
        raise KeyError(
            "Expected `initial_conv.weight` in the upstream OpenSpliceAI checkpoint to apply the "
            "nucleotide input-channel conversion, but it is absent."
        )
    state_dict: OrderedDict = OrderedDict()
    for key, value in original_state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue
        new_key = convert_original_state_dict_key(key, config)
        state_dict[new_key] = convert_original_state_dict_value(key, value, new_vocab)
    return state_dict


def convert_checkpoint(convert_config):
    print(f"Converting OpenSpliceAI checkpoint at {convert_config.checkpoint_path}")
    context = _infer_context(convert_config.checkpoint_path, convert_config.context)
    config = _make_config(context)

    model = Model(config)

    model_alphabet = get_alphabet("nucleobase", prepend_tokens=[])
    new_vocab = list(model_alphabet)

    # weights_only=False is required: some OpenSpliceAI releases pickle the full model object, not a state dict.
    original = torch.load(convert_config.checkpoint_path, map_location="cpu", weights_only=False)
    if hasattr(original, "state_dict"):
        original = original.state_dict()

    state_dict = _convert_checkpoint(original, config, new_vocab)

    model_state = model.state_dict()
    for key, value in model_state.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    load_checkpoint(model, state_dict)

    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def convert_all_checkpoints(convert_config):
    variants = _iter_variants()
    missing = [
        _checkpoint_path(convert_config.checkpoint_root, species, context, convert_config.seed)
        for species, context in variants
        if not Path(_checkpoint_path(convert_config.checkpoint_root, species, context, convert_config.seed)).exists()
    ]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing OpenSpliceAI checkpoints:\n{missing_list}")

    if convert_config.repo_id is not None:
        raise ValueError(
            "Do not pass repo_id with convert_all; each variant writes to its own multimolecule/<variant> repo."
        )

    for species, context in variants:
        child = ConvertConfig()
        child.root = convert_config.root
        child.output_path = _variant_name(species, context)
        child.checkpoint_root = convert_config.checkpoint_root
        child.checkpoint_path = _checkpoint_path(convert_config.checkpoint_root, species, context, convert_config.seed)
        child.context = context
        child.species = species
        child.seed = convert_config.seed
        child.default_variant = convert_config.default_variant
        child.push_to_hub = convert_config.push_to_hub
        child.delete_existing = convert_config.delete_existing
        child.validate_golden = convert_config.validate_golden
        child.delete_after_validate = convert_config.delete_after_validate
        child.golden_root = convert_config.golden_root
        child.token = convert_config.token
        child.repo_id = f"multimolecule/{child.output_path}"
        convert_checkpoint(child)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str | None = None  # type: ignore[assignment]
    checkpoint_path: str | None = None  # type: ignore[assignment]
    checkpoint_root: str = "models"
    context: int | None = None
    species: str = "mane"
    seed: int = DEFAULT_SEED
    convert_all: bool = False
    default_variant: str | None = DEFAULT_VARIANT

    def post(self):
        if self.convert_all:
            return
        if self.checkpoint_path is None and self.context is None:
            self.context = 10000
        context = _infer_context(self.checkpoint_path, self.context)
        if self.output_path is None:
            self.output_path = _variant_name(self.species, context)
        if self.checkpoint_path is None:
            self.checkpoint_path = _checkpoint_path(self.checkpoint_root, self.species, context, self.seed)
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    if config.convert_all:
        convert_all_checkpoints(config)
    else:
        convert_checkpoint(config)
