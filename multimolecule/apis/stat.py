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

import os
import shutil
from statistics import mean
from typing import List

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


def get_result_stat(experiment_root: str, remove_empty: bool = True) -> List[Result]:
    results = []
    for root, _, files in tqdm(os.walk(experiment_root)):
        if "run.log" in files:
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
            seed = config.seed
            result = Result(id=best.id, pretrained=pretrained, seed=seed)
            result.validation = NestedDict(
                {k: format(mean(v) if isinstance(v, list) else v, ".8f") for k, v in best.validation.all_items()}
            )
            result.test = NestedDict(
                {k: format(mean(v) if isinstance(v, list) else v, ".8f") for k, v in best.test.all_items()}
            )
            result.epoch = best.index
            result.pop("validation.time", None)
            result.pop("test.time", None)
            result.pop("validation.loss", None)
            result.pop("test.loss", None)
            result.pop("validation.lr", None)
            result.pop("test.lr", None)
            results.append(result)
    # Remove empty directories, perform twice to remove all empty directories
    if remove_empty:
        for root, dirs, files in os.walk(experiment_root):
            if not files and not dirs:
                os.rmdir(root)
        for root, dirs, files in os.walk(experiment_root):
            if not files and not dirs:
                os.rmdir(root)
    results.sort(key=lambda x: (x.pretrained, x.seed, x.id))
    return results


def write_result_stat(results: List[Result], path: str):
    results = [dict(result.all_items()) for result in results]  # type: ignore[misc]
    df = pd.DataFrame.from_dict(results)
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
    if not len(result_stat) > 0:
        raise ValueError("No results found")
    write_result_stat(result_stat, config.out_path)
