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

from functools import wraps
from typing import Callable

import pandas as pd
import psycopg2
from psycopg2 import InterfaceError, OperationalError

CONN_STRING = "host='hh-pgsql-public.ebi.ac.uk' dbname='pfmegrnargs' user='reader' password='NWDMCE5xdipIjRrp'"


def connect(conn_string: str = CONN_STRING) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn = kwargs.pop("conn", None)
            while True:
                try:
                    if conn is None:
                        conn = psycopg2.connect(conn_string)
                    kwargs["conn"] = conn
                    return func(*args, **kwargs)
                except (OperationalError, InterfaceError):
                    print("Connection lost. Reconnecting...")
                    conn = psycopg2.connect(conn_string)

        return wrapper

    return decorator


@connect(conn_string=CONN_STRING)
def execute(query, filter: Callable | None = None, batch_size: int = 1_000_000, eager_break: bool = True, conn=None):
    offset = 0
    data = None
    query = query.replace(";", "")

    while True:
        with conn.cursor() as cursor:
            cursor.execute(f"{query} LIMIT %s OFFSET %s", [batch_size, offset])
            batch = cursor.fetchall()
            batch_len = len(batch)
            if not batch:
                break
            batch = pd.DataFrame.from_dict({desc[0]: col for desc, col in zip(cursor.description, zip(*batch))})
            if filter is not None:
                batch = filter(batch)
            data = pd.concat([data, batch]) if data is not None else batch
            offset += len(batch)
            print(f"Fetched {offset} data so far...")
            if eager_break and batch_len < batch_size:
                break

    return data
