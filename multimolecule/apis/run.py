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

# mypy: disable-error-code="attr-defined"

import atexit
import os
import warnings
from typing import Type

import danling as dl
import torch

from multimolecule.runners import MultiMoleculeConfig, MultiMoleculeRunner

try:
    import nni
except ImportError:
    nni = None


def train(
    config: MultiMoleculeConfig = None,  # type: ignore
    runner_cls: Type[MultiMoleculeRunner] = MultiMoleculeRunner,
):
    if config is None:
        config = MultiMoleculeConfig()
    config = config.parse(default_config="config", no_default_config_action="warn")
    config.interpolate(unsafe_eval=True)
    config.training = True
    if config.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if config.get("nni", False):
        if nni is None:
            raise ValueError("Unable to retrieve nni parameters, since nni is not installed.")
        config.merge(nni.get_next_parameter())
    with dl.debug(config.get("debug", False)):
        runner = runner_cls(config)
        atexit.register(runner.print_result)
        atexit.register(runner.save_result)
        atexit.register(runner.save_checkpoint)
        result = runner.train()
        return result


def evaluate(
    config: MultiMoleculeConfig = None,  # type: ignore
    runner_cls: Type[MultiMoleculeRunner] = MultiMoleculeRunner,
):
    if config is None:
        config = MultiMoleculeConfig.empty()
    config = config.parse(default_config="config", no_default_config_action="warn")
    config.interpolate(unsafe_eval=True)
    config.training = False
    if config.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if "checkpoint" not in config or not isinstance(config.checkpoint, str):
        raise RuntimeError("Please specify `checkpoint` to run evaluate")
    for name, data in config.datas.items():
        if "evaluation" not in data or not isinstance(data.evaluate, str):
            raise RuntimeError(f"Please specify `evaluation` to run evaluate in datas.{name}")
    runner = runner_cls(config)
    result = runner.evaluate_epoch("evaluation")
    print(result)
    return result


def infer(
    config: MultiMoleculeConfig = None,  # type: ignore
    runner_cls: Type[MultiMoleculeRunner] = MultiMoleculeRunner,
):
    if config is None:
        config = MultiMoleculeConfig.empty()
    config = config.parse(default_config="config", no_default_config_action="warn")
    config.interpolate(unsafe_eval=True)
    config.training = False
    if config.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if "checkpoint" not in config or not isinstance(config.checkpoint, str):
        raise RuntimeError("Please specify `checkpoint` to run infer.")
    for name, data in config.datas.items():
        if "inference" not in data or not isinstance(data.inference, str):
            raise RuntimeError(f"Please specify `inference` to run infer in datas.{name}")
    if "result_path" not in config or not isinstance(config.result_path, str):
        config.result_path = os.path.join(os.getcwd(), "result.json")
        warnings.warn("`result_path` is not specified, default to `result.json`.", RuntimeWarning, stacklevel=2)
    runner = runner_cls(config)
    result = runner.infer()
    runner.save(result, config.result_path)
    return result
