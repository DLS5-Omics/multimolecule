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
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from sqlalchemy.exc import OperationalError

from multimolecule.datasets import database
from multimolecule.datasets.database import (
    DatabaseConfig,
    DatabaseConnection,
    close_connections,
    get_config_path,
    get_connection,
    query,
    query_dataset,
)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_database.yaml"
    path.write_text(
        yaml.safe_dump({"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}),
        encoding="utf-8",
    )
    return path


class TestDatabaseConfig:
    def test_config_creation_with_all_fields(self, tmp_path: Path):
        config_path = tmp_path / "test_database.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "dialect": "postgresql",
                    "driver": "psycopg2",
                    "host": "localhost",
                    "database": "testdb",
                    "user": "testuser",
                    "password": "testpass",
                    "port": 5432,
                }
            ),
            encoding="utf-8",
        )

        config = DatabaseConfig.from_yaml(str(config_path))

        assert config.dialect == "postgresql"
        assert config.host == "localhost"
        assert config.database == "testdb"
        assert config.user == "testuser"
        assert config.password == "testpass"
        assert config.port == 5432

    def test_config_with_empty_password(self, tmp_path: Path):
        config_path = tmp_path / "test_database.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "dialect": "mysql",
                    "driver": "pymysql",
                    "host": "public-db.example.com",
                    "database": "publicdb",
                    "user": "readonly",
                    "password": "",
                    "port": 3306,
                }
            ),
            encoding="utf-8",
        )

        config = DatabaseConfig.from_yaml(str(config_path))

        assert config.build_url() == "mysql+pymysql://readonly:@public-db.example.com:3306/publicdb"

    def test_build_url_postgresql(self):
        config = DatabaseConfig(
            dialect="postgresql",
            driver="psycopg2",
            host="localhost",
            database="testdb",
            user="user",
            password="pass",
            port=5432,
        )

        assert config.build_url() == "postgresql+psycopg2://user:pass@localhost:5432/testdb"

    def test_build_url_mysql(self):
        config = DatabaseConfig(
            dialect="mysql",
            driver="pymysql",
            host="mysql.example.com",
            database="mydb",
            user="myuser",
            password="mypass",
            port=3306,
        )

        assert config.build_url() == "mysql+pymysql://myuser:mypass@mysql.example.com:3306/mydb"

    def test_build_url_missing_required_fields(self):
        config = DatabaseConfig(dialect="postgresql", host="localhost")

        with pytest.raises(ValueError, match="Missing database config fields") as exc_info:
            config.build_url()

        assert "database" in str(exc_info.value)
        assert "user" in str(exc_info.value)

    def test_direct_url_override(self):
        config = DatabaseConfig(url="postgresql://direct:url@host:5432/db", dialect="mysql", host="ignored")

        assert config.build_url() == "postgresql://direct:url@host:5432/db"

    def test_pool_config_defaults(self):
        config = DatabaseConfig(dialect="postgresql", host="localhost", database="db", user="user")

        pool_config = config.get_pool_config()

        assert pool_config["pool_size"] == 5
        assert pool_config["max_overflow"] == 10
        assert pool_config["pool_pre_ping"]

    def test_pool_config_override(self):
        config = DatabaseConfig(
            dialect="postgresql",
            host="localhost",
            database="db",
            user="user",
            pool={"pool_size": 10, "max_overflow": 20},
        )

        pool_config = config.get_pool_config()

        assert pool_config["pool_size"] == 10
        assert pool_config["max_overflow"] == 20


class TestDatabaseConnection:
    def test_connection_creation(self, config_path: Path):
        conn = DatabaseConnection(str(config_path))

        assert conn.config is not None
        assert conn.config.dialect == "sqlite"

    def test_normalize_query(self):
        assert DatabaseConnection._normalize_query("SELECT  *   FROM   table   WHERE  id = 1  ;") == (
            "SELECT * FROM table WHERE id = 1"
        )
        assert DatabaseConnection._normalize_query("SELECT * FROM table;") == "SELECT * FROM table"
        assert DatabaseConnection._normalize_query("SELECT * FROM table   ") == "SELECT * FROM table"

    def test_extract_limit(self, config_path: Path):
        conn = DatabaseConnection(str(config_path))

        assert conn._extract_limit("SELECT * FROM table LIMIT 100") == ("SELECT * FROM table", 100)
        assert conn._extract_limit("SELECT * FROM table LIMIT 50 OFFSET 10") == ("SELECT * FROM table", 50)
        assert conn._extract_limit("SELECT * FROM table") == ("SELECT * FROM table", None)

    def test_extract_limit_negative(self, config_path: Path):
        conn = DatabaseConnection(str(config_path))

        with pytest.raises(ValueError, match="LIMIT cannot be negative"):
            conn._extract_limit("SELECT * FROM table LIMIT -1")

    def test_retry_mechanism(self, monkeypatch, config_path: Path):
        conn = DatabaseConnection(str(config_path))
        monkeypatch.setattr(database, "create_engine", Mock(return_value=Mock()))
        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OperationalError("Connection failed", None, None)
            return "success"

        assert conn._retry(failing_function) == "success"
        assert call_count == 3

    def test_retry_exhausted(self, monkeypatch, config_path: Path):
        conn = DatabaseConnection(str(config_path))
        conn._max_retries = 2
        monkeypatch.setattr(database, "create_engine", Mock(return_value=Mock()))

        def always_failing_function():
            raise OperationalError("Always fails", None, None)

        with pytest.raises(OperationalError):
            conn._retry(always_failing_function)


class TestQueryFunctions:
    def test_query_function(self, monkeypatch, config_path: Path):
        mock_conn = Mock()
        mock_df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        mock_conn.query.return_value = mock_df
        mock_get_connection = Mock(return_value=mock_conn)
        monkeypatch.setattr(database, "get_connection", mock_get_connection)

        result = query(str(config_path), "SELECT * FROM test")

        assert isinstance(result, pd.DataFrame)
        mock_get_connection.assert_called_once_with(str(config_path))
        mock_conn.query.assert_called_once()

    def test_query_dataset_function(self, monkeypatch, config_path: Path):
        mock_get_config_path = Mock(return_value=str(config_path))
        mock_query = Mock(return_value=pd.DataFrame({"result": [42]}))
        monkeypatch.setattr(database, "get_config_path", mock_get_config_path)
        monkeypatch.setattr(database, "query", mock_query)

        result = query_dataset("test_dataset", "SELECT 42 as result")

        assert isinstance(result, pd.DataFrame)
        mock_get_config_path.assert_called_once_with("test_dataset", None)
        mock_query.assert_called_once()

    def test_get_config_path(self):
        assert get_config_path("test_dataset").endswith("test_dataset/database.yaml")
        assert get_config_path("test_dataset", "/custom/path") == "/custom/path/test_dataset/database.yaml"

    def test_connection_pooling(self, config_path: Path, tmp_path: Path):
        close_connections()

        conn1 = get_connection(str(config_path))
        conn2 = get_connection(str(config_path))

        assert conn1 is conn2

        second_config_path = tmp_path / "test_database2.yaml"
        second_config_path.write_text(
            yaml.safe_dump({"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}),
            encoding="utf-8",
        )

        assert get_connection(str(second_config_path)) is not conn1

    def test_close_connections(self, config_path: Path):
        conn1 = get_connection(str(config_path))

        close_connections()

        assert get_connection(str(config_path)) is not conn1


class TestErrorHandling:
    def test_empty_query_validation(self, config_path: Path):
        conn = DatabaseConnection(str(config_path))

        with pytest.raises(ValueError, match="Query cannot be empty"):
            conn.query("")

    def test_invalid_batch_size(self, config_path: Path):
        conn = DatabaseConnection(str(config_path))

        with pytest.raises(ValueError, match="Batch size must be positive"):
            conn.query("SELECT 1", batch_size=0)

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            DatabaseConnection("/non/existent/path.yaml")
