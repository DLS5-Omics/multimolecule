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
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import torch
from tqdm import tqdm

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset
from multimolecule.io import read_ct

torch.manual_seed(1016)


@contextmanager
def preprocess_ct(file: Path) -> Iterator[Path]:
    lines = [line for line in file.read_text().splitlines() if line.strip()]
    first_line = lines[0].strip().split()
    num_bases = int(first_line[0])
    changed = False

    for i in range(1, num_bases + 1):
        if i >= len(lines) or int(lines[i].split()[0]) != i:
            next_index = i + 1 if i < num_bases else 0
            lines.insert(i, f"{i} N {i - 1} {next_index} 0 {i}")
            changed = True

        parts = lines[i].split()
        if int(parts[4]) == -1:
            parts[4] = "0"
            lines[i] = " ".join(parts)
            changed = True

    if not changed:
        yield file
        return

    with TemporaryDirectory() as directory:
        corrected = Path(directory) / file.name
        corrected.write_text("\n".join(lines) + "\n")
        yield corrected


def convert_ct(file, family: str) -> Mapping:
    if not isinstance(file, Path):
        file = Path(file)
    with preprocess_ct(file) as corrected:
        record = read_ct(corrected)

    parts = list(file.parts)
    parts = parts[parts.index(family + "_database") :]
    parts[0] = parts[0][:-9]
    parts[-1] = parts[-1][:-3]

    return {
        "id": "-".join(parts),
        "sequence": record.sequence,
        "secondary_structure": record.dot_bracket,
        "family": family,
        "subfamily": parts[1] if len(parts) == 3 else None,
    }


def _file_sort_key(file: str | Path, root: Path) -> tuple[str, int]:
    relative = Path(file).relative_to(root).as_posix()
    letters = "".join(filter(str.isalpha, relative))
    digits = "".join(filter(str.isdigit, relative))
    return letters, int(digits or 0)


def _convert_dataset(family_dir, max_seq_len: int | None = None):
    family_dir = Path(family_dir)
    family = family_dir.stem[:-9]
    files = [os.path.join(family_dir, f) for f in os.listdir(family_dir) if f.endswith(".ct")]
    if not files:
        for subdir in family_dir.iterdir():
            if subdir.is_dir():
                files.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".ct")])
    files.sort(key=lambda file: _file_sort_key(file, family_dir))
    data = [convert_ct(file, family) for file in tqdm(files, total=len(files))]
    if max_seq_len is not None:
        data = [d for d in data if len(d["sequence"]) <= max_seq_len]
    return data


def convert_dataset(convert_config):
    max_seq_len = convert_config.max_seq_len
    families = [
        os.path.join(convert_config.dataset_path, f)
        for f in os.listdir(convert_config.dataset_path)
        if f.endswith("_database")
    ]
    families.sort()
    data = [i for family in families for i in _convert_dataset(family, max_seq_len)]
    save_dataset(convert_config, data, filename="train.parquet")


class ConvertConfig(ConvertConfig_):
    max_seq_len: int | None = None
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__))

    def post(self):
        if self.max_seq_len is not None:
            self.output_path = f"{self.output_path}.{self.max_seq_len}"
        super().post()


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
