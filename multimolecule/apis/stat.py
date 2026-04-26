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
import shutil
from statistics import mean

import chanfig
import pandas as pd
from chanfig import NestedDict
from tqdm import tqdm


class Result(NestedDict):
    pretrained: str
    id: str
    seed: int
    epoch: int
    validation: NestedDict
    test: NestedDict


def get_result_stat(experiment_root: str, remove_empty: bool = True) -> list[Result]:
    results = []
    for root, _, files in tqdm(os.walk(experiment_root)):
        if "run.log" not in files:
            continue
        if "best.json" not in files:
            if remove_empty:
                shutil.rmtree(root)
            continue
        best = NestedDict.from_json(os.path.join(root, "best.json"))
        if "index" not in best:
            if remove_empty:
                shutil.rmtree(root)
            continue
        config = NestedDict.from_yaml(os.path.join(root, "trainer.yaml"))
        pretrained = config.pretrained.split("/")[-1]
        result = Result(id=best.id, pretrained=pretrained, seed=config.seed)
        validation = best.get("validation", best.get("val", NestedDict()))
        test = best.get("test", NestedDict())
        result.validation = NestedDict(
            {
                key: format(mean(value) if isinstance(value, list) else value, ".8f")
                for key, value in validation.all_items()
            }
        )
        result.test = NestedDict(
            {key: format(mean(value) if isinstance(value, list) else value, ".8f") for key, value in test.all_items()}
        )
        result.epoch = best.index
        for key in ("validation.time", "test.time", "validation.loss", "test.loss", "validation.lr", "test.lr"):
            result.pop(key, None)
        results.append(result)
    if remove_empty:
        for root, dirs, files in os.walk(experiment_root):
            if not files and not dirs:
                os.rmdir(root)
        for root, dirs, files in os.walk(experiment_root):
            if not files and not dirs:
                os.rmdir(root)
    results.sort(key=lambda x: (x.pretrained, x.seed, x.id))
    return results


def write_result_stat(results: list[Result], path: str) -> None:
    rows = [dict(result.all_items()) for result in results]
    df = pd.DataFrame.from_dict(rows)
    df.insert(len(df.keys()) - 1, "comment", "")
    df.fillna("")
    df.to_csv(path, index=False)


class Config(chanfig.Config):
    experiment_root: str = "experiments"
    out_path: str = "result.csv"
    remove_empty: bool = True


if __name__ == "__main__":
    config = Config().parse()
    result_stat = get_result_stat(config.experiment_root, config.remove_empty)
    if not result_stat:
        raise ValueError("No results found")
    write_result_stat(result_stat, config.out_path)
