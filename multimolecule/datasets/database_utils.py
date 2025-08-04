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

import re
import time
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chanfig
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as SQLTimeoutError

_LIMIT_PATTERN = re.compile(r"\s+LIMIT\s+(-?\d+)(?:\s+OFFSET\s+\d+)?\s*$", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

_DEFAULT_POOL_CONFIG = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "pool_timeout": 30,
    "echo": False,
}

_connection_pool: Dict[str, DatabaseConnection] = {}


class DatabaseConfig(chanfig.NestedDict):
    dialect: str
    host: str
    database: str
    user: str
    password: str
    driver: str = ""
    port: Optional[int] = None
    url: Optional[str] = None
    pool: Optional[Dict[str, Any]] = None

    def build_url(self) -> str:
        if self.url:
            return self.url

        required = ["dialect", "host", "database", "user"]
        missing = [f for f in required if not getattr(self, f, None)]
        if missing:
            raise ValueError(f"Missing database config fields: {missing}")

        # Password can be empty for public databases
        password = getattr(self, "password", "") or ""

        dialect = self.dialect.lower().strip()
        driver = f"+{self.driver.strip()}" if self.driver else ""
        port = f":{self.port}" if self.port else ""

        return f"{dialect}{driver}://{self.user.strip()}:{password}@{self.host.strip()}{port}/{self.database.strip()}"

    def get_pool_config(self) -> Dict[str, Any]:
        config = _DEFAULT_POOL_CONFIG.copy()
        if self.pool:
            config.update(self.pool)
        return config


class DatabaseConnection:
    def __init__(self, config_path: str):
        self.config = DatabaseConfig.from_yaml(config_path)
        self._engine: Optional[Engine] = None
        self._max_retries = 3
        self._retry_delay = 1.0

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            url = self.config.build_url()
            pool_config = self.config.get_pool_config()
            self._engine = create_engine(url, **pool_config)
        return self._engine

    @staticmethod
    def _normalize_query(query: str) -> str:
        return _WHITESPACE_PATTERN.sub(" ", query.strip().rstrip(";")).strip()

    def _extract_limit(self, query: str) -> tuple[str, Optional[int]]:
        match = _LIMIT_PATTERN.search(query)
        if match:
            try:
                limit = int(match.group(1))
                if limit < 0:
                    raise ValueError("LIMIT cannot be negative")
                return _LIMIT_PATTERN.sub("", query).strip(), limit
            except (ValueError, IndexError) as e:
                if "LIMIT cannot be negative" in str(e):
                    raise
                raise ValueError(f"Invalid LIMIT clause: {e}")
        return query, None

    def _retry(self, func: Callable, max_retries: int | None = None) -> Any:
        max_retries = max_retries or self._max_retries
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func()
            except (DisconnectionError, OperationalError, SQLTimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self._retry_delay * (2**attempt)
                    time.sleep(delay)
                    self._engine = None
                    continue
                break
            except SQLAlchemyError:
                raise

        raise last_exception or RuntimeError("Database operation failed after retries")

    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        filter_func: Optional[Callable] = None,
        batch_size: int = 1_000_000,
        eager_break: bool = True,
    ) -> pd.DataFrame:
        if not sql or not sql.strip():
            raise ValueError("Query cannot be empty")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        sql = self._normalize_query(sql)
        params = params or {}
        base_query, user_limit = self._extract_limit(sql)

        dataframes: List[pd.DataFrame] = []
        offset = 0
        paginated_query = f"{base_query} LIMIT :batch_size OFFSET :offset"

        while True:
            effective_batch_size = batch_size
            if user_limit is not None:
                remaining = user_limit - offset
                if remaining <= 0:
                    break
                effective_batch_size = min(batch_size, remaining)

            query_params = {**params, "batch_size": effective_batch_size, "offset": offset}

            def execute_batch(params_dict, query_template):
                with self.engine.connect() as conn:
                    result = conn.execute(text(query_template), params_dict)
                    rows = result.fetchall()
                    if not rows:
                        return None, 0
                    return pd.DataFrame(rows, columns=result.keys()), len(rows)

            batch_result = self._retry(partial(execute_batch, query_params, paginated_query))
            if batch_result is None or batch_result[0] is None:
                break

            batch_df, batch_len = batch_result

            if filter_func is not None:
                try:
                    batch_df = filter_func(batch_df)
                    if batch_df is None or len(batch_df) == 0:
                        offset += batch_len
                        continue
                except Exception as e:
                    raise ValueError(f"Filter function failed: {e}")

            dataframes.append(batch_df)
            offset += batch_len

            if eager_break and batch_len < effective_batch_size:
                break

        if not dataframes:
            return pd.DataFrame()
        elif len(dataframes) == 1:
            return dataframes[0].reset_index(drop=True)
        else:
            return pd.concat(dataframes, ignore_index=True, copy=False)

    def test(self) -> bool:
        try:

            def test_query():
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    return True

            return self._retry(test_query, max_retries=1)
        except Exception:
            return False

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def connect_database(config_path: str) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            db_conn = kwargs.pop("db_conn", None)
            if db_conn is None:
                db_conn = DatabaseConnection(config_path)
            kwargs["db_conn"] = db_conn
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_config_path(dataset_name: str, datasets_root: Optional[str] = None) -> str:
    if datasets_root is None:
        datasets_root = str(Path(__file__).parent)
    return str(Path(datasets_root) / dataset_name / "database.yaml")


def get_connection(config_path: str) -> DatabaseConnection:
    if config_path not in _connection_pool:
        _connection_pool[config_path] = DatabaseConnection(config_path)
    return _connection_pool[config_path]


def close_connections() -> None:
    for conn in _connection_pool.values():
        conn.close()
    _connection_pool.clear()


def query(
    config_path: str,
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    filter_func: Optional[Callable] = None,
    batch_size: int = 1_000_000,
    eager_break: bool = True,
) -> pd.DataFrame:
    return get_connection(config_path).query(sql, params, filter_func, batch_size, eager_break)


def query_dataset(
    dataset_name: str,
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    filter_func: Optional[Callable] = None,
    batch_size: int = 1_000_000,
    eager_break: bool = True,
    datasets_root: Optional[str] = None,
) -> pd.DataFrame:
    config_path = get_config_path(dataset_name, datasets_root)
    return query(config_path, sql, params, filter_func, batch_size, eager_break)


# Backward compatibility aliases
execute_query = query
execute_query_by_dataset = query_dataset
get_dataset_config_path = get_config_path
close_all_connections = close_connections
