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
import shutil

import pyarrow as pa
from chanfig import Config
from pandas import DataFrame
from pyarrow import Table

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def get_files(path: str) -> list[str]:
    files = [os.path.join(path, f) for f in os.listdir(path)]
    files.sort(key=lambda f: ("".join(filter(str.isalpha, f)), int("".join(filter(str.isdigit, f)))))
    return files


def write_data(
    data: Table | list | dict | DataFrame,
    output_path: str,
    filename: str = "data.parquet",
    compression: str = "brotli",
    level: int = 4,
):
    if isinstance(data, list):
        data = Table.from_pylist(data)
    elif isinstance(data, dict):
        data = Table.from_pydict(data)
    elif isinstance(data, DataFrame):
        data = Table.from_pandas(data)
    if not isinstance(data, Table):
        raise ValueError("Data must be a list, dict, pandas DataFrame, or pyarrow Table.")

    pa.parquet.write_table(data, os.path.join(output_path, filename), compression=compression, compression_level=level)


def copy_readme(root: str, output_path: str):
    readme = f"README.{output_path}.md" if f"README.{output_path}.md" in os.listdir(root) else "README.md"
    shutil.copy2(os.path.join(root, readme), output_path)


def push_to_hub(convert_config: ConvertConfig, output_path: str, repo_type: str = "dataset"):
    if convert_config.push_to_hub:
        if HfApi is None:
            raise ImportError("Please install huggingface_hub to push to the hub.")
        api = HfApi()
        if convert_config.delete_existing:
            api.delete_repo(convert_config.repo_id, token=convert_config.token, missing_ok=True)
        api.create_repo(convert_config.repo_id, token=convert_config.token, exist_ok=True, repo_type=repo_type)
        api.upload_folder(
            repo_id=convert_config.repo_id, folder_path=output_path, token=convert_config.token, repo_type=repo_type
        )


def save_dataset(
    convert_config: ConvertConfig, data: Table | list | dict | DataFrame, compression: str = "brotli", level: int = 4
):
    root, output_path = convert_config.root, convert_config.output_path
    os.makedirs(output_path, exist_ok=True)
    write_data(data, output_path, compression=compression, level=level)
    copy_readme(root, output_path)
    push_to_hub(convert_config, output_path)


class ConvertConfig(Config):
    dataset_path: str
    root: str
    output_path: str
    push_to_hub: bool = False
    delete_existing: bool = False
    repo_id: str | None = None
    token: str | None = None

    def post(self):
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"
