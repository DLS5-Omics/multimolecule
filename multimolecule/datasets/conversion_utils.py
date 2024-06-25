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

from chanfig import Config
from pandas import DataFrame

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def save_dataset(convert_config: ConvertConfig, dataframe: DataFrame):
    root, output_path = convert_config.root, convert_config.output_path
    os.makedirs(output_path, exist_ok=True)

    dataframe.to_parquet(os.path.join(output_path, "data.parquet"))

    if f"README.{output_path}.md" in os.listdir(root):
        shutil.copy2(os.path.join(root, f"README.{output_path}.md"), output_path)
    else:
        shutil.copy2(os.path.join(root, "README.md"), output_path)

    if convert_config.push_to_hub:
        if HfApi is None:
            raise ImportError("Please install huggingface_hub to push to the hub.")
        api = HfApi()
        if convert_config.delete_existing:
            api.delete_repo(convert_config.repo_id, token=convert_config.token, missing_ok=True)
        api.create_repo(convert_config.repo_id, token=convert_config.token, exist_ok=True, repo_type="dataset")
        api.upload_folder(
            repo_id=convert_config.repo_id, folder_path=output_path, token=convert_config.token, repo_type="dataset"
        )


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
