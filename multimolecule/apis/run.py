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

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import atexit
import importlib.util
import os
import warnings

import danling as dl
import torch
from art import text2art
from lazy_imports import try_import

from multimolecule.runner import RUNNERS, Config

with try_import() as nni_import:
    import nni


def dynamic_import(pretrained: str | None = None):
    import sys

    cwd = os.getcwd()
    sys.path.append(os.path.dirname(cwd))
    importlib.import_module(os.path.basename(cwd))
    if pretrained is not None and pretrained in os.listdir(cwd):
        sys.path.append(cwd)
        importlib.import_module(pretrained)


def train(
    config: Config = None,  # type: ignore[assignment]
):
    print(text2art("Multi\nMolecule", "rand-large"))
    if config is None:
        config = Config()
    config = config.parse(default_config="config", no_default_config_action="warn")
    config.interpolate(unsafe_eval=True)
    dynamic_import(config.pretrained)
    if config.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if config.get("nni", False):
        nni_import.check()
        config.merge(nni.get_next_parameter())
    with dl.debug(config.get("debug", False)):
        runner = RUNNERS.build(config)
        print(runner)
        atexit.register(runner.print_result)
        atexit.register(runner.save_result)
        atexit.register(runner.save_checkpoint)
        result = runner.train()
        return result


def evaluate(
    config: Config = None,  # type: ignore[assignment]
):
    print(text2art("Multi\nMolecule", "rand-large"))
    if config is None:
        config = Config.empty()
    config = config.parse(default_config="config", no_default_config_action="warn")
    config.training = False
    config.interpolate(unsafe_eval=True)
    dynamic_import(config.pretrained)
    if config.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    # if "checkpoint" not in config or not isinstance(config.checkpoint, str):
    #    raise RuntimeError("Please specify `checkpoint` to run evaluate")
    # if "data" in config:
    #    if not config.data.get("evaluate"):  # type: ignore[union-attr]
    #        raise RuntimeError("Please specify `evaluate` to run evaluate.")
    # elif "datas" in config:
    #     for name, data in config.datas.items():
    #         if not data.get("evaluate"):
    #             raise RuntimeError(f"Please specify `evaluate` to run evaluate in datas.{name}")
    runner = RUNNERS.build(config)
    print(runner)
    result = runner.evaluate()
    print(result)
    return result


def infer(
    config: Config = None,  # type: ignore[assignment]
):
    print(text2art("Multi\nMolecule", "rand-large"))
    if config is None:
        config = Config.empty()
    config = config.parse(default_config="config", no_default_config_action="warn")
    config.training = False
    config.interpolate(unsafe_eval=True)
    dynamic_import(config.pretrained)
    if config.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    # if "checkpoint" not in config or not isinstance(config.checkpoint, str):
    #     raise RuntimeError("Please specify `checkpoint` to run infer.")
    # if "data" in config:
    #     if not config.data.get("infer"):  # type: ignore[union-attr]
    #         raise RuntimeError("Please specify `infer` to run infer.")
    # elif "datas" in config:  # type: ignore[union-attr]
    #     for name, data in config.datas.items():
    #         if not data.get("infer"):
    #             raise RuntimeError(f"Please specify `infer` to run infer in datas.{name}")
    if "result_path" not in config:
        config.result_path = os.path.join(os.getcwd(), "result.json")
        warnings.warn("`result_path` is not specified, default to `result.json`.", RuntimeWarning, stacklevel=2)
    runner = RUNNERS.build(config)
    print(runner)
    result = runner.infer()
    runner.save(result, config.result_path)
    return result
