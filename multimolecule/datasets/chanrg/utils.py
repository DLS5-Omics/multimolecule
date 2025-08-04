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

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from ..database_utils import connect_database

# Get the config path for this dataset
_CONFIG_PATH = str(Path(__file__).parent / "database.yaml")


@connect_database(_CONFIG_PATH)
def query(
    query: str,
    filter: Callable | None = None,
    batch_size: int = 1_000_000,
    eager_break: bool = True,
    params: Optional[Dict[str, Any]] = None,
    db_conn=None,
) -> pd.DataFrame:
    return db_conn.query(query, params, filter, batch_size, eager_break)
