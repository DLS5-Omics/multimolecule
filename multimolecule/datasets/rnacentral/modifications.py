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

import torch

from multimolecule.datasets.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.datasets.conversion_utils import save_dataset
from multimolecule.datasets.rnacentral.utils import execute

torch.manual_seed(1016)

command = """
WITH active_rna_precomputed AS (
    SELECT *
    FROM rnc_rna_precomputed
    WHERE is_active = True AND taxid IS NULL
)
SELECT
    rna.upi AS urs,
    rna.id AS rna_id,
    rna.seq_short AS sequence,
    rnc_modifications.id AS modification_id,
    rnc_modifications.modification_id AS modification_type,
    rnc_modifications.position AS modification_position,
    rnc_modifications.accession AS modification_accession,
    active_rna_precomputed.rna_type,
    active_rna_precomputed.so_rna_type
FROM
    rnc_modifications
JOIN
    rna
ON
    rnc_modifications.upi = rna.upi
JOIN
    active_rna_precomputed
ON
    rnc_modifications.upi = active_rna_precomputed.upi
"""


def convert_dataset(config: ConvertConfig):
    df = execute(command)
    df = df.groupby("urs").agg(lambda x: list(x) if x.name.startswith("modification") else x.iloc[0])
    df.sort_values(["urs", "rna_id"], inplace=True)
    df.reset_index(inplace=True)
    modification_columns = [col for col in df.keys() if col.startswith("modification")]
    modifications = df.apply(
        lambda row: {
            k: {col[13:]: row[col][i] for col in modification_columns if col != "modification_id"}
            for i, k in sorted(enumerate(row["modification_id"]), key=lambda x: x[1])
        },
        axis=1,
    )
    df.insert(3, "modifications", modifications)
    df = df.drop(columns=modification_columns)
    save_dataset(config, df)


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = os.path.basename(os.path.dirname(__file__)) + "-df"


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_dataset(config)
