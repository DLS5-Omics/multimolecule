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
from typing import Dict

from chanfig import Config
from transformers import PreTrainedModel
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from multimolecule.tokenisers.rna.utils import get_tokenizer_config

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def write_model(
    output_path: str,
    model: PreTrainedModel,
    tokenizer_config: Dict | None = None,
):
    model.save_pretrained(output_path, safe_serialization=True)
    model.save_pretrained(output_path, safe_serialization=False)
    if tokenizer_config is None:
        tokenizer_config = get_tokenizer_config()
        tokenizer_config["model_max_length"] = getattr(model.config, "max_position_embeddings", None)
    tokenizer = tokenizer_class_from_name(tokenizer_config["tokenizer_class"])(**tokenizer_config)
    tokenizer.save_pretrained(output_path)


def copy_readme(root: str, output_path: str):
    readme = f"README.{output_path}.md" if f"README.{output_path}.md" in os.listdir(root) else "README.md"
    shutil.copy2(os.path.join(root, readme), os.path.join(output_path, "README.md"))


def push_to_hub(convert_config: ConvertConfig, output_path: str, repo_type: str = "model"):
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


def save_checkpoint(
    convert_config: ConvertConfig,
    model: PreTrainedModel,
    tokenizer_config: Dict | None = None,
):
    root, output_path = convert_config.root, convert_config.output_path
    write_model(output_path, model, tokenizer_config)
    copy_readme(root, output_path)
    push_to_hub(convert_config, output_path)


class ConvertConfig(Config):
    checkpoint_path: str
    root: str
    output_path: str
    push_to_hub: bool = False
    delete_existing: bool = False
    repo_id: str | None = None
    token: str | None = None

    def post(self):
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"
