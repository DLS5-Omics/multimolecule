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

import importlib
import os
import sys
import warnings
from typing import Any, cast

import danling as dl
import torch
from lazy_imports import try_import

from multimolecule.runner import RUNNERS, Config

with try_import() as art_import:
    from art import text2art

with try_import() as nni_import:
    import nni


def dynamic_import(pretrained: str | None = None) -> None:
    r"""
    Import the working-directory package (and an optional `pretrained` sub-directory) so user-defined datasets,
    models, and runners register themselves before the runner is built.

    Appends the parent of `cwd` to `sys.path` and imports `os.path.basename(cwd)` as a Python module. If
    `pretrained` matches a sub-directory of `cwd`, that sub-directory is also appended and imported.

    Args:
        pretrained: Optional secondary module to import — typically the experiment's `pretrained` identifier
            when it lives under `cwd`.
    """
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    if parent not in sys.path:
        sys.path.append(parent)
    importlib.import_module(os.path.basename(cwd))
    if pretrained is not None and pretrained in os.listdir(cwd):
        if cwd not in sys.path:
            sys.path.append(cwd)
        importlib.import_module(pretrained)


def print_banner() -> None:
    if art_import.is_successful():
        print(text2art("Multi\nMolecule", "rand-large"))
    else:
        print("MultiMolecule")


def prepare_config(config: Config | None = None, *, training: bool | None = None) -> Config:
    r"""
    Parse, finalize, and apply runtime side-effects to a runner [`Config`][multimolecule.runner.Config].

    Reads `config.yaml` from the current working directory (warning if missing) and applies CLI overrides via
    [`Config.parse`][danling.RunnerConfig], then forces `config.training` if requested, runs the
    [`dynamic_import`][multimolecule.api.run.dynamic_import] hook, applies CUDA precision flags
    (`allow_tf32`, `reduced_precision_reduction`), and merges NNI hyperparameters when `config.nni` is truthy.

    Args:
        config: Optional pre-built configuration. When `None`, a fresh `Config()` is created and populated
            entirely from `config.yaml` plus CLI arguments.
        training: When set, overrides `config.training` after parsing — used by the `mmtrain` / `mmevaluate` /
            `mminfer` entry points to flip training mode without requiring a YAML edit.

    Returns:
        The parsed and finalized configuration.
    """
    if config is None:
        config = Config()
    parsed_config = config.parse(default_config="config", no_default_config_action="warn")
    if parsed_config is None:
        raise ValueError("Failed to parse config")
    config = cast(Config, parsed_config)
    if training is not None:
        config.training = training
    config.interpolate()
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
    return config


def train(config: Config | None = None) -> Any:
    r"""
    Train a model with `config.training = True`.

    Wraps the run in a [`dl.debug`][danling.debug] context when `config.debug` is truthy, then invokes
    `runner.train()`. Console-script entry point for `mmtrain`.

    Args:
        config: Optional pre-built configuration; CLI overrides are still applied via
            [`prepare_config`][multimolecule.api.run.prepare_config].

    Returns:
        Whatever the underlying [`Runner.train`][multimolecule.runner.Runner] returns.
    """
    print_banner()
    config = prepare_config(config, training=True)
    with dl.debug(config.get("debug", False)):
        runner = RUNNERS.build(config)
        print(runner)
        return runner.train()


def evaluate(config: Config | None = None) -> Any:
    r"""
    Evaluate a model on the configured splits with `config.training = False`.

    Console-script entry point for `mmevaluate`.

    Args:
        config: Optional pre-built configuration; CLI overrides are still applied.

    Returns:
        Whatever the underlying [`Runner.evaluate`][multimolecule.runner.Runner] returns.
    """
    print_banner()
    config = prepare_config(config, training=False)
    runner = RUNNERS.build(config)
    print(runner)
    result = runner.evaluate()
    print(result)
    return result


def infer(config: Config | None = None) -> Any:
    r"""
    Run inference and save predictions to `config.result_path` (defaults to `./result.json` with a warning).

    Console-script entry point for `mminfer`.

    Args:
        config: Optional pre-built configuration; CLI overrides are still applied.

    Returns:
        Whatever the underlying [`Runner.infer`][multimolecule.runner.Runner] returns.
    """
    print_banner()
    config = prepare_config(config, training=False)
    if "result_path" not in config:
        config.result_path = os.path.join(os.getcwd(), "result.json")
        warnings.warn("`result_path` is not specified, defaulting to `result.json`.", RuntimeWarning, stacklevel=2)
    runner = RUNNERS.build(config)
    print(runner)
    result = runner.infer()
    runner.save(result, config.result_path)
    return result


__all__ = ["dynamic_import", "evaluate", "infer", "prepare_config", "train"]
